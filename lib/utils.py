import os
from pathlib import Path
from tqdm import tqdm

# 获取当前运行环境
in_colab = False
num_workers = 0
try:
    import google.colab as colab  # 在Colab上

    in_colab = True
    num_workers = 2
except:
    pass

# 路径相关配置
root_path = Path(__file__).parents[1]  # 项目根目录
data_path = root_path / "data"  # 数据目录
models_path = root_path / "models"  # 模型目录


# 从kaggle下载数据
# 若不提供path，默认下载到data目录下
# 返回下载文件的路径
# 【Warning】由于kaggle api的限制，没有获取真实的文件名。这里单纯地将{key}.zip作为了文件名。不知道会不会出现错误。
def kaggle_download(key, path=data_path):
    import subprocess
    # 如果是 Colab 上，检查kaggle是否配置
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if in_colab and not os.path.exists(kaggle_path):
        colab.drive.mount("/content/drive", force_remount=True)
        Path(kaggle_path).parent.mkdir(parents=True, exist_ok=True)
        from shutil import copyfile
        copyfile("/content/drive/MyDrive/kaggle.json", kaggle_path)
    subprocess.call(["kaggle", "competitions", "download", "-c", key, "-p", path])
    return os.path.join(path, f"{key}.zip")


# 如果不提供解压路径，则解压到新文件夹里
def unzip(file, targetdir=None):
    from zipfile import ZipFile
    if targetdir is None:
        targetdir = Path(file).parent
    Path(targetdir).mkdir(parents=True, exist_ok=True)

    with ZipFile(file) as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Unzip"):
            zip_ref.extract(member=file, path=targetdir)


# 从kaggle下载数据并解压
# 文件解压后，放到data/dir_name中（也就是说只能放到data中。为了方便放弃了一定的自由度）
# 若不提供dir_name，则与key同名。
# 返回解压目录
def kaggle_download_extract(key, dir_name=None):
    if dir_name is None:
        dir_name = key
    zipfile = kaggle_download(key, data_path)  # 临时下载到data目录
    unzip(zipfile, os.path.join(data_path, dir_name))
    os.remove(zipfile)


# 定时函数。如果距离上次此函数返回true时间超过t秒，就返回true。第一次调用返回true
_last_t = 0.0
def time_passed(t):
    global _last_t
    import time
    now = time.time()
    if now - _last_t >= t:
        _last_t = now
        return True
    else:
        return False


# 定距函数。每调用n次就返回一次true。
_loop_counter = 0
def loop_passed(n):
    global _loop_counter
    if _loop_counter < n:
        _loop_counter += 1
        return False
    else:
        _loop_counter = 0
        return True
