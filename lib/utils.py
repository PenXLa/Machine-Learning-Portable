import os
from pathlib import Path

# 获取当前运行环境
in_colab = False
try:
    import google.colab as colab  # 在Colab上
    in_colab = True
except: pass

# 路径相关配置
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # 项目根目录
data_path = os.path.join(root_path, "data") # 数据目录
models_path = os.path.join(root_path, "models") # 模型目录


# 从kaggle下载数据
# 若不提供path，默认下载到data目录下
# 返回下载文件的路径
# 【Warning】由于kaggle api的限制，没有获取真实的文件名。这里单纯地将{key}.zip作为了文件名。不知道会不会出现错误。
def kaggle_download(key, path=data_path):
    import subprocess
    # 如果是 Colab 上，检查kaggle是否配置
    if in_colab and not os.path.exists("~/.kaggle/kaggle.json"):
        colab.drive.mount("/content/drive", force_remount=True)
        Path("～/.kaggle").mkdir(parents=True, exist_ok=True)
        from shutil import copyfile
        copyfile("/content/drive/MyDrive/kaggle.json", "~/.kaggle/kaggle.json")
    subprocess.call(["kaggle", "competitions", "download", "-c", key, "-p", path])
    return os.path.join(path, f"{key}.zip")

# 如果不提供解压路径，则解压到新文件夹里
def unzip(zipfile, targetdir=None):
    import zipfile
    if targetdir is None:
        targetdir = os.path.splitext(zipfile)[0]
    Path(targetdir).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zipfile, "r") as zip_ref:
        zip_ref.extractall(targetdir)

# 从kaggle下载数据并解压
# 文件解压后，放到data/dir_name中（也就是说只能放到data中。为了方便放弃了一定的自由度）
# 若不提供dir_name，则与key同名。
# 返回解压目录
def kaggle_download_extract(key, dir_name=None):
    zipfile = kaggle_download(key, data_path) # 临时下载到data目录
    unzip(zipfile, os.path.join(data_path, dir_name))
    os.remove(zipfile)