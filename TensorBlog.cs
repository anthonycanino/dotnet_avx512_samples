using System.Numerics;
using System.Numerics.Tensors;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;

using BenchmarkDotNet.Running;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;

namespace Avx512Blog
{
  public class TensorBlog
  {
    // dimensionality for text-embedding-ada-002 and text-search-davinci-*-001
    [Params(10, 100, 1000, 10000)]
    public int Dimensionality;

    private float[] input; // cannot have this be a `Span` (this class is not a ref struct)
    private float[] input2;
    private List< float[] > docsCollection = new();

    [GlobalSetup]
    public void Setup()
    {
      input = GenerateRandom(Dimensionality);
      input2 = GenerateRandom(Dimensionality);
    }

    [Benchmark]
    public void SimilarityScalar()
    {
      CosineSimilarity(input, input2);
    }

    [Benchmark]
    public void SimilarityScalarVec()
    {
      CosineSimilarityVec(input, input2);
    }

    [Benchmark]
    public void SimilarityFromSysNum()
    {
      TensorPrimitives.CosineSimilarity(input, input2);
    }

    private static float[] GenerateRandom(int size)
    {
      return Enumerable.Range(0,size).Select(_ => Random.Shared.NextSingle()).ToArray();
    }

    private static float CosineSimilarity(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
      if (x.Length != y.Length) {
        throw new ArgumentException("Array lengths must be equal");
      }
      float dot = 0, xSumSquared = 0, ySumSquared = 0;

      for (int i = 0; i < x.Length; i++) {
        dot += x[i] * y[i];
        xSumSquared += x[i] * x[i];
        ySumSquared += y[i] * y[i];
      }

      return dot / (MathF.Sqrt(xSumSquared) * MathF.Sqrt(ySumSquared));
    }

    // from https://github.com/microsoft/semantic-kernel/blob/3451a4ebbc9db0d049f48804c12791c681a326cb/dotnet/src/SemanticKernel.Core/AI/Embeddings/VectorOperations/CosineSimilarityOperation.cs#L119
    private static unsafe float CosineSimilarityVec(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
      if (x.Length != y.Length) {
        throw new ArgumentException("Array lengths must be equal");
      }
      fixed (float* pxBuffer = x, pyBuffer = y) {
        double dotSum = 0, lenXSum = 0, lenYSum = 0;

        float* px = pxBuffer, py = pyBuffer;
        float* pxEnd = px + x.Length;

        if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count) {
          float* pxOneVectorFromEnd = pxEnd - Vector<float>.Count;
          do {
            Vector<float> xVec = *(Vector<float>*)px;
            Vector<float> yVec = *(Vector<float>*)py;

            dotSum += Vector.Dot(xVec, yVec); // Dot product
            lenXSum += Vector.Dot(xVec, xVec); // For magnitude of x
            lenYSum += Vector.Dot(yVec, yVec); // For magnitude of y

            px += Vector<float>.Count;
            py += Vector<float>.Count;
          } while (px <= pxOneVectorFromEnd);
        }

        while (px < pxEnd) {
          float xVal = *px;
          float yVal = *py;

          dotSum += xVal * yVal; // Dot product
          lenXSum += xVal * xVal; // For magnitude of x
          lenYSum += yVal * yVal; // For magnitude of y

          ++px;
          ++py;
        }

        // Cosine Similarity of X, Y
        // Sum(X * Y) / |X| * |Y|
        return (float)(dotSum / (Math.Sqrt(lenXSum) * Math.Sqrt(lenYSum)));
      }
    }

    private static unsafe float CosineSimilarityVec512(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
      if (x.Length != y.Length) {
        throw new ArgumentException("Array lengths must be equal");
      }
      fixed (float* pxBuffer = x, pyBuffer = y) {
        double dotSum = 0, lenXSum = 0, lenYSum = 0;

        float* px = pxBuffer, py = pyBuffer;
        float* pxEnd = px + x.Length;

        if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<float>.Count) {
          float* pxOneVectorFromEnd = pxEnd - Vector512<float>.Count;
          do {
            Vector512<float> xVec = *(Vector512<float>*)px;
            Vector512<float> yVec = *(Vector512<float>*)py;

            dotSum += Vector512.Dot(xVec, yVec); // Dot product
            lenXSum += Vector512.Dot(xVec, xVec); // For magnitude of x
            lenYSum += Vector512.Dot(yVec, yVec); // For magnitude of y

            px += Vector512<float>.Count;
            py += Vector512<float>.Count;
          } while (px <= pxOneVectorFromEnd);
        }

        while (px < pxEnd) {
          float xVal = *px;
          float yVal = *py;

          dotSum += xVal * yVal; // Dot product
          lenXSum += xVal * xVal; // For magnitude of x
          lenYSum += yVal * yVal; // For magnitude of y

          ++px;
          ++py;
        }

        // Cosine Similarity of X, Y
        // Sum(X * Y) / |X| * |Y|
        return (float)(dotSum / (Math.Sqrt(lenXSum) * Math.Sqrt(lenYSum)));
      }
    }
  }
}