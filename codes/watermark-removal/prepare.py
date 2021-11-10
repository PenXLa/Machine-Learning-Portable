from lib.utils import *
from shutil import copyfile


copyfile(data_drive / "watermark-removal/CLWD.rar", data_path / "watermark-removal/CLWD.rar")
extractAll(data_path / "watermark-removal/CLWD.rar", delete = True)
