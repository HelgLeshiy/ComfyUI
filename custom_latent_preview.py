import functools
import logging
import os
import uuid
from PIL import Image
import time
import io
import struct
from threading import Thread
import torch.nn.functional as F
import torch

import latent_preview
import server

serv = server.PromptServer.instance


def hook(obj, attr):
    logging.info(f"Hooking {obj} {attr}")

    def dec(f):
        logging.info(f"Wrapping {f}")
        f = functools.update_wrapper(f, getattr(obj, attr))
        setattr(obj, attr, f)
        return f

    return dec


rates_table = {
    "Mochi": 24 // 6,
    "LTXV": 24 // 8,
    "HunyuanVideo": 24 // 4,
    "Cosmos1CV8x8x8": 24 // 8,
    "Wan21": 16 // 4,
    "Wan22": 24 // 4,
}


class WrappedPreviewer(latent_preview.LatentPreviewer):
    def __init__(self, previewer, rate=8):
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        if hasattr(previewer, "taesd"):
            self.taesd = previewer.taesd
        elif hasattr(previewer, "latent_rgb_factors"):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
            self.latent_rgb_factors_reshape = getattr(
                previewer, "latent_rgb_factors_reshape", None
            )
        else:
            raise Exception("Unsupported preview type for VHS animated previews")

    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            # Keep batch major
            x0 = x0.movedim(2, 1)
            x0 = x0.reshape((-1,) + x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews / self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            serv.send_sync(
                "VHS_latentpreview",
                {"length": num_images, "rate": self.rate, "id": serv.last_node_id},
            )
            self.last_time = new_time + 1 / self.rate
        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index : self.c_index + num_previews]

        filename = self.process_previews(x0, self.c_index, num_images)
        self.c_index = (self.c_index + num_previews) % num_images
        return filename

    def process_previews(self, image_tensor, ind, leng):
        image_tensor = self.decode_latent_to_preview(image_tensor)
        if image_tensor.size(1) > 512 or image_tensor.size(2) > 512:
            image_tensor = image_tensor.movedim(-1, 0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (512 * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(
                    image_tensor, (height, 512), mode="bilinear"
                )
            else:
                width = (512 * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(
                    image_tensor, (512, width), mode="bilinear"
                )
            image_tensor = image_tensor.movedim(0, -1)
        previews_ubyte = (
            ((image_tensor + 1.0) / 2.0)
            .clamp(0, 1)  # change scale from -1..1 to 0..1
            .mul(0xFF)  # to 0..255
        ).to(device="cpu", dtype=torch.uint8)

        gif = []

        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            gif.append(i)
            # message = io.BytesIO()
            # message.write((1).to_bytes(length=4, byteorder='big')*2)
            # message.write(ind.to_bytes(length=4, byteorder='big'))
            # message.write(struct.pack('16p', serv.last_node_id.encode('ascii')))
            # i.save(message, format="JPEG", quality=95, compress_level=1)
            # #NOTE: send sync already uses call_soon_threadsafe
            # serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
            #                message.getvalue(), serv.client_id)
            ind = (ind + 1) % leng
        filename = str(uuid.uuid4()) + ".gif"
        gif[0].save(
            os.path.join("/outputs", filename),
            save_all=True,
            optimize=False,
            append_images=gif[1:],
            loop=0,
        )
        return filename

    def decode_latent_to_preview(self, x0):
        if hasattr(self, "taesd"):
            x_sample = self.taesd.decode(x0).movedim(1, 3)
            return x_sample
        else:
            if self.latent_rgb_factors_reshape is not None:
                x0 = self.latent_rgb_factors_reshape(x0)
            self.latent_rgb_factors = self.latent_rgb_factors.to(
                dtype=x0.dtype, device=x0.device
            )
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(
                    dtype=x0.dtype, device=x0.device
                )
            latent_image = F.linear(
                x0.movedim(1, -1),
                self.latent_rgb_factors,
                bias=self.latent_rgb_factors_bias,
            )
            return latent_image


@hook(latent_preview, "get_previewer")
def get_latent_video_previewer(device, latent_format, *args, **kwargs):
    logging.info(f"Call for wrapped get previewver {device} {latent_format}")
    node_id = serv.last_node_id
    base = get_latent_video_previewer.__wrapped__(device, latent_format, *args, **kwargs)

    # If Comfy has previews disabled, but we still want our animated previews:
    if base is None:
        logging.info("Base get_previewer returned None (preview method probably 'none'). Forcing Latent2RGB previewer for VHS.")
        try:
            # Newer Latent2RGBPreviewer takes (latent_rgb_factors, latent_rgb_factors_bias)
            base = latent_preview.Latent2RGBPreviewer(
                latent_format.latent_rgb_factors,
                getattr(latent_format, "latent_rgb_factors_bias", None),
            )
        except Exception as e:
            logging.warning(f"Failed to construct fallback previewer: {e}")
            return None
    try:
        rate_setting = rates_table.get(latent_format.__class__.__name__, 8)
    except:
        # For safety since there's lots of keys, any of which can fail
        logging.warning("Failed to init settings for previewer")

    logging.info("Successfully returning WrapperPreviewer")
    return WrappedPreviewer(base, rate_setting)
