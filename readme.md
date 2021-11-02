* codes：各种ML实验所在的目录。
* lib：常用的工具函数放到lib里，方便共享调用。
* models：预训练模型和训练好的模型
* data：存放数据的目录。

PyCharm支持直接import项目任意文件。在Colab上可能存在找不到module的情况。最好把项目根目录加到Python path里。

lib.utils 中定义了 `root_drive` 等以 drive 结尾的路径。这些是用于数据持久化的路径。
在 Colab 上时，路径指向 Google Drive；在本地上时，路径指向项目本身。