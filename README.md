# mamecog

まめコグ － C#用の小さなCNNエンジン

## 概要

まめコグ（mamecog）は、Kerasで構築したCNN（Convolutional Neural Network）の学習済みモデルを用いて、C#アプリケーションでCNNの推論を行うツールです。まめコグにはCNNの学習機能は含みません。

まめコグの最大の特徴は、小さな4個のC#クラス（LayerData2D, Conv2D, MaxPool2D, Dense）だけでCNNライブラリが構成されていることです。まめコグでは、Kerasの学習済みモデルをC#で読み込み可能な独自形式に変換し、これを用いてC#アプリケーションでCNNの推論を実行します。他のC#ライブラリへの依存もなく、様々なC#アプリケーションでCNNを利用可能とすることを目指しています。

## 使用方法

Keras公式ページの[Simple MNIST convnet（https://keras.io/examples/vision/mnist_convnet/）](https://keras.io/examples/vision/mnist_convnet/)をサンプルとして、まめコグC#ライブラリを用いてCNNを実行するための手順を説明します。

### 【手順１】Kerasで学習済みモデルを保存する

上記URLのサンプルでfit()メソッドの後に下記のコードを追加し、学習済みモデルを.h5ファイルとして保存します。

```
model.save("my_model.h5")
```

このとき、Kerasのsummary()メソッドで学習済みモデルの構造を確認しておきます。今回のMNISTサンプルでは次のようなサマリーとなります。

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                16010     
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________
```

### 【手順２】Kerasの学習済みモデルをまめコグ形式に変換する

まめコグでは、CNNを用いたC#アプリケーションを開発する前に、Kerasで構築したCNNの学習済みファイルを、まめコグC#ライブラリの各クラスで使用する独自形式のバイナリファイルに変換します。
手順１で作成したmy_model.h5ファイルを入力として、下記のように変換ツール（mamecog_converter.py）を実行します。

```
python mamecog_converter.py my_model.h
```

MNISTサンプルでは、mamecog_converter.pyは下記の6個のbinファイルを出力します。

```
1_conv2d_b.bin    1_conv2d_k.bin
3_conv2d_1_b.bin  3_conv2d_1_k.bin
7_dense_b.bin     7_dense_k.bin
```

これらのファイルには、手順１のKerasモデルのconv2d層、conv2d_1層、dense層の学習済みのカーネルとバイアスが、まめコグC#ライブラリ用に変換されて格納されています。

### 【手順３】C#で学習済みモデルを読み込む

次のようにConv2DとDenseのインスタンスを生成し、手順２のbinファイルを読み込みます。このとき、手順１で作成したCNNモデルの各層のサイズを指定します。

```
Conv2D conv1 = new Conv2D(32, 1, 3, 3);  //出力Ch数、入力Ch数、カーネル縦サイズ、カーネル横サイズ
Conv2D conv2 = new Conv2D(64, 32, 3, 3);
Dense dense = new Dense(10, 1600);       //出力Ch数、入力Ch数
conv1.LoadKernelAndBias("1_conv2d_k.bin", "1_conv2d_b.bin");
conv2.LoadKernelAndBias("3_conv2d_1_k.bin", "3_conv2d_1_b.bin");
dense.LoadKernelAndBias("7_dense_k.bin", "7_dense_b.bin");
```

### 【手順４】C#で各層のデータを保存する領域を用意する

Conv2DとMaxPool2Dの入出力を格納するためのLayerData2Dのインスタンスを作成します。Dense層の出力は1次元配列に格納します。

```
LayerData2D input0 = new LayerData2D(1, 28, 28);
LayerData2D conv1output = new LayerData2D(32, 26, 26);
LayerData2D pool1output = new LayerData2D(32, 13, 13);
LayerData2D conv2output = new LayerData2D(64, 11, 11);
LayerData2D pool2output = new LayerData2D(64, 5, 5);
float[] pool2flatten = new float[64 * 5 * 5];
float[] denseOutput = new float[10];
```

### 【手順５】C#でCNNの推論を実行する

下記のように各層の出力を順に計算します。

```
conv1.Calc(conv1output, input0);
MaxPool2D.Calc(pool1output, conv1output, 2);
conv2.Calc(conv2output, pool1output);
MaxPool2D.Calc(pool2output, conv2output, 2);
pool2output.Flatten(pool2flatten);
dense.Calc(denseOutput, pool2flatten);
```

## 動作環境および開発環境

下記の環境で開発と動作確認を行っています。
- Windows 10 Pro (64bit)
- Visual Studio Community 2019
- .Net Framework 4.7.2

CNNの学習済みモデルを下記の環境で構築しています。
- Python 3.7.8
- TensorFlow 2.2.0

## 開発状況と今後の予定

現状（初期バージョン）では、Kerasでモデルを構築する際にConv2D層などのパラメーターにデフォルト値を指定したときにのみ対応しています。例えば、「パディングあり」のモデルには未対応です。また、Conv2D層の活性化関数はReLUのみ、Dense層の活性化関数はSoftmaxのみに対応しています。

今後のバージョンで下記の機能追加を予定しています。
- 「パディングあり」等のKerasのConv2Dクラスのデフォルト以外のオプションのサポート
- ReLUとSoftmax以外の活性化関数のサポート
- マルチコア活用等の高速化
- .Net Coreのサポート
- Xamarinでの動作確認
- VGG16等の動作サンプルの追加

更に、C++版の開発も予定しています。

## ライセンス

mamecog（まめコグ）は、MITライセンスで公開します。
"as is"（現状のまま）の提供です。一切の保証はありません。
ご使用は自己責任でお願いします。

## 開発者

mamecog（まめコグ）の開発者は、Hideki Hashimoto（84moto）です。
ご連絡は[https://twitter.com/hashimov](https://twitter.com/hashimov)にお願いします。

