修改请提pr

### 这是建模比赛D题的Demo

如果想运行代码，请安装以下库：

```sh
pip install scipy
pip install numpy
pip install matplotlib
pip install scikit-learn
```

图片示例为main分支下的.png文件，运行时可以将绘图代码部分注释来实现效果

### Update
2024.6.6 分位数赋法以及平均值调整

2024.6.6 使用高斯混合模型优化数据并使用随机生成a值

2024.6.7 调整了a值的范围，并对生成的数据根据平均值进行调整，同时所有上下限调整均采用重采样而不是clip()

2024.6.7 初稿完成

2024.6.16 直接使用分位数赋法

### Todo
