import hashlib
from os import environ, makedirs, replace
from os.path import exists, isdir, join
from shutil import copyfileobj
import tarfile
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder


class _BerkeleyRedirects(HTTPRedirectHandler):
    """Keep dataset downloads on the authenticated publisher origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urlsplit(newurl)
        if url.scheme != "https" or url.netloc.lower() != "www2.eecs.berkeley.edu":
            raise ValueError("Untrusted BSDS300 redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300", "images")
    if exists(output_image_dir):
        return output_image_dir

    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("This dataset loader requires Python with tarfile.data_filter support.")

    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    makedirs(dest, exist_ok=True)
    with TemporaryDirectory(dir=dest) as staging:
        archive = join(staging, "BSDS300-images.tgz")
        print("Downloading BSDS300 from", url)
        with build_opener(_BerkeleyRedirects()).open(url, timeout=30) as data:
            with open(archive, "wb") as output:
                copyfileobj(data, output)

        # Consumers can pin the publisher archive to their independently verified digest.
        expected_digest = environ.get("BSDS300_SHA256")
        if expected_digest:
            digest = hashlib.sha256()
            with open(archive, "rb") as data:
                for block in iter(lambda: data.read(65536), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_digest.lower():
                raise ValueError("BSDS300 archive checksum mismatch")

        unpacked = join(staging, "unpacked")
        makedirs(unpacked)
        with tarfile.open(archive) as bundle:
            for item in bundle:
                if not (item.isfile() or item.isdir()):
                    raise ValueError("Unexpected entry type in the BSDS300 archive")
                bundle.extract(item, unpacked, filter="data")

        for split in ("train", "test"):
            if not isdir(join(unpacked, "BSDS300", "images", split)):
                raise ValueError("Missing BSDS300 image split: " + split)
        replace(join(unpacked, "BSDS300"), join(dest, "BSDS300"))
    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
