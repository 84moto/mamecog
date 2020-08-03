//
// Conv2D.cs
// Copyright © 2020 Hideki Hashimoto
//
// https://github.com/84moto/mamecog
//
// This software is released under the MIT License.
//

// 現在のバージョンは、KerasのConv2Dクラスのオプションのうち、
// 下記を選択した場合のみをサポートする。
// （他のオプションは今後サポートを追加予定）
// strides=(1, 1)
// padding="valid"
// data_format=None
// dilation_rate=(1, 1)
// groups=1
// use_bias=True
// kernel_regularizer=None
// bias_regularizer=None
// activity_regularizer=None
// kernel_constraint=None
// bias_constraint=None

// 活性化関数はReLUのみ実装 

using System;
using System.IO;

namespace Mamecog
{
    /// <summary>
    /// Conv2D計算クラス
    /// </summary>
    public class Conv2D
    {
        public int OutputPlaneNum { get; }
        public int InputPlaneNum { get; }
        public int KernelHeight { get; }
        public int KernelWidth { get; }
        public float[] Kernel;
        public float[] Bias;

        /// <summary>
        /// カーネルの形状を指定してConv2D計算用オブジェクトを作成する
        /// </summary>
        /// <param name="outputPlaneNum">出力層の面の数</param>
        /// <param name="inputPlaneNum">入力層の面の数</param>
        /// <param name="kernelHeight">カーネスサイズ（縦）</param>
        /// <param name="kernelWidth">カーネスサイズ（横）</param>
        public Conv2D(int outputPlaneNum, int inputPlaneNum, int kernelHeight, int kernelWidth)
        {
            OutputPlaneNum = outputPlaneNum;
            InputPlaneNum = inputPlaneNum;
            KernelHeight = kernelHeight;
            KernelWidth = kernelWidth;
            Kernel = new float[outputPlaneNum * inputPlaneNum * kernelHeight * kernelWidth];
            Bias = new float[outputPlaneNum];
        }

        /// <summary>
        /// ファイルからカーネルとバイアスの値を読み込む
        /// </summary>
        /// <param name="kernelFileName">カーネルデータのファイル名</param>
        /// <param name="biasFileName">バイアスデータのファイル名</param>
        public void LoadKernelAndBias(string kernelFileName, string biasFileName)
        {
            using (Stream stream = File.OpenRead(kernelFileName))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    for (int i = 0; i < Kernel.Length; i++)
                    {
                        float f = reader.ReadSingle();
                        Kernel[i] = f;
                    }
                }
            }
            using (Stream stream = File.OpenRead(biasFileName))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    for (int i = 0; i < Bias.Length; i++)
                    {
                        float f = reader.ReadSingle();
                        Bias[i] = f;
                    }
                }
            }
        }

        public float GetKernelVal(int outputPlane, int inputPlane, int kernelY, int kernelX)
        {
            int idx = InputPlaneNum * KernelHeight * KernelWidth * outputPlane
                    + KernelHeight * KernelWidth * inputPlane
                    + KernelWidth * kernelY + kernelX;
            return Kernel[idx];
        }

        /// <summary>
        /// （デバッグ用）カーネルとバイアスの値を表示する
        /// </summary>
        public void PrintKernelAndBias()
        {
            Console.WriteLine("Number of Output Planes = " + OutputPlaneNum.ToString());
            Console.WriteLine("Number of Input Planes = " + InputPlaneNum.ToString());
            for (int outputPlane = 0; outputPlane < OutputPlaneNum; outputPlane++)
            {
                for (int inputPlane = 0; inputPlane < InputPlaneNum; inputPlane++)
                {
                    Console.WriteLine("Kernel " + inputPlane.ToString() + " -> " + outputPlane.ToString());
                    for (int kernelY = 0; kernelY < KernelHeight; kernelY++)
                    {
                        for (int kernelX = 0; kernelX < KernelWidth; kernelX++)
                        {
                            float w = GetKernelVal(outputPlane, inputPlane, kernelY, kernelX);
                            Console.Write(w.ToString() + ", ");
                        }
                        Console.Write("\n");
                    }
                }
            }
            Console.WriteLine("Bias");
            for (int outputPlane = 0; outputPlane < OutputPlaneNum; outputPlane++)
            {
                float w = Bias[outputPlane];
                Console.Write(w.ToString() + ", ");
            }
            Console.Write("\n");
        }

        /// <summary>
        /// Conv2D層の計算を行う
        /// </summary>
        /// <param name="outputLayer">出力層のデータを格納するLayerData2Dオブジェクト</param>
        /// <param name="inputLayer">入力層のデータが格納されたLayerData2Dオブジェクト</param>
        public void Calc(LayerData2D outputLayer, LayerData2D inputLayer)
        {
            if (inputLayer.PlaneNum != InputPlaneNum)
                throw new Exception("InputPlaneNum不整合");
            if (outputLayer.PlaneNum != OutputPlaneNum)
                throw new Exception("OutputPlaneNum不整合");
            if (inputLayer.PlaneWidth - KernelWidth / 2 - KernelWidth / 2 != outputLayer.PlaneWidth)
                throw new Exception("Planeサイズ不整合");
            if (inputLayer.PlaneHeight - KernelHeight / 2 - KernelHeight / 2 != outputLayer.PlaneHeight)
                throw new Exception("Planeサイズ不整合");

            for (int outputPlane = 0; outputPlane < outputLayer.PlaneNum; outputPlane++)
            {
                int outputPlaneStartIdx = outputLayer.PlaneHeight * outputLayer.PlaneWidth * outputPlane;
                for (int outputY = 0; outputY < outputLayer.PlaneHeight; outputY++)
                {
                    int outputRowStartIdx = outputPlaneStartIdx + outputLayer.PlaneWidth * outputY;
                    for (int outputX = 0; outputX < outputLayer.PlaneWidth; outputX++)
                    {
                        int outputCellIdx = outputRowStartIdx + outputX;
                        float sum = 0;
                        for (int inputPlane = 0; inputPlane < inputLayer.PlaneNum; inputPlane++)
                        {
                            int inputPlaneStartIdx = inputLayer.PlaneHeight * inputLayer.PlaneWidth * inputPlane;
                            int kernelStartIdx = InputPlaneNum * KernelHeight * KernelWidth * outputPlane + KernelHeight * KernelWidth * inputPlane;
                            for (int kernelY = 0; kernelY < KernelHeight; kernelY++)
                            {
                                int kernelRowStartIdx = kernelStartIdx + KernelWidth * kernelY;
                                int inputY = outputY + kernelY;
                                int inputRowStartIdx = inputPlaneStartIdx + inputLayer.PlaneWidth * inputY;
                                for (int kernelX = 0; kernelX < KernelWidth; kernelX++)
                                {
                                    //sum += Weights[outputPlane, inputPlane, kernelY, kernelX] * input.Cells[inputPlane, inputY, inputX];
                                    //w = Weights[InputPlaneNum * KernelHeight * KernelWidth * outputPlane + KernelHeight * KernelWidth * inputPlane + KernelWidth * kernelY + kernelX]
                                    int kernelIdx = kernelRowStartIdx + kernelX;
                                    int inputX = outputX + kernelX;
                                    int inputCellIdx = inputRowStartIdx + inputX;
                                    sum += Kernel[kernelIdx] * inputLayer.Cells[inputCellIdx];
                                }
                            }
                        }
                        sum += Bias[outputPlane];

                        //活性化関数（ReLU）
                        float cellOutput = ReLU(sum);
                        outputLayer.Cells[outputCellIdx] = cellOutput;
                    }
                }
            }
        }

        private float ReLU(float sum)
        {
            if (sum > 0)
                return sum;
            else
                return 0.0F;
        }
    }
}
