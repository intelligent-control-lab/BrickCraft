"""List connected 3Dconnexion HID devices and their hidraw paths.

Run:
    conda activate brickcraft
    python -m brickcraft.teleop.list_spacemouse
"""

SUPPORTED_VENDOR_IDS = {0x256F, 0x046D}


def main() -> None:
    from easyhid import Enumeration

    devices = [
        device
        for device in Enumeration().find()
        if device.vendor_id in SUPPORTED_VENDOR_IDS
    ]
    if not devices:
        print("No 3Dconnexion HID devices found.")
        return
    for device in devices:
        path = device.path.decode() if isinstance(device.path, bytes) else device.path
        print(
            f"{path}  vendor=0x{device.vendor_id:04x} "
            f"product=0x{device.product_id:04x}  {device.product_string}"
        )


if __name__ == "__main__":
    main()
