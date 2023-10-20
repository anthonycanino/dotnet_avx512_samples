using System.Numerics;
using System.Numerics.Tensors;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

using BenchmarkDotNet.Running;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;

using System.Buffers;
using System.Buffers.Text;

using SixLabors.ImageSharp.Memory;

namespace Avx512Blog
{
  public class ColorConversionBlog
  {

        public static int Precision = 32;
        public static float MaximumValue = MathF.Pow(2, Precision) - 1;
        public static float HalfValue = MathF.Ceiling(MaximumValue * 0.5F);   // /2

#pragma warning disable SA1206 // Declaration keywords should follow order
    public readonly ref struct ComponentValues
#pragma warning restore SA1206 // Declaration keywords should follow order
    {
        /// <summary>
        /// The component count
        /// </summary>
        public readonly int ComponentCount;

        /// <summary>
        /// The component 0 (eg. Y)
        /// </summary>
        public readonly Span<float> Component0;

        /// <summary>
        /// The component 1 (eg. Cb). In case of grayscale, it points to <see cref="Component0"/>.
        /// </summary>
        public readonly Span<float> Component1;

        /// <summary>
        /// The component 2 (eg. Cr). In case of grayscale, it points to <see cref="Component0"/>.
        /// </summary>
        public readonly Span<float> Component2;

        /// <summary>
        /// The component 4
        /// </summary>
        public readonly Span<float> Component3;

        /// <summary>
        /// Initializes a new instance of the <see cref="ComponentValues"/> struct.
        /// </summary>
        /// <param name="componentBuffers">List of component buffers.</param>
        /// <param name="row">Row to convert</param>
        public ComponentValues(IReadOnlyList<Buffer2D<float>> componentBuffers, int row)
        {
            this.ComponentCount = componentBuffers.Count;

            this.Component0 = componentBuffers[0].DangerousGetRowSpan(row);

            // In case of grayscale, Component1 and Component2 point to Component0 memory area
            this.Component1 = this.ComponentCount > 1 ? componentBuffers[1].DangerousGetRowSpan(row) : this.Component0;
            this.Component2 = this.ComponentCount > 2 ? componentBuffers[2].DangerousGetRowSpan(row) : this.Component0;
            this.Component3 = this.ComponentCount > 3 ? componentBuffers[3].DangerousGetRowSpan(row) : Span<float>.Empty;
        }

        internal ComponentValues(
            int componentCount,
            Span<float> c0,
            Span<float> c1,
            Span<float> c2,
            Span<float> c3)
        {
            this.ComponentCount = componentCount;
            this.Component0 = c0;
            this.Component1 = c1;
            this.Component2 = c2;
            this.Component3 = c3;
        }

        public ComponentValues Slice(int start, int length)
        {
            Span<float> c0 = this.Component0.Slice(start, length);
            Span<float> c1 = this.Component1.Length > 0 ? this.Component1.Slice(start, length) : Span<float>.Empty;
            Span<float> c2 = this.Component2.Length > 0 ? this.Component2.Slice(start, length) : Span<float>.Empty;
            Span<float> c3 = this.Component3.Length > 0 ? this.Component3.Slice(start, length) : Span<float>.Empty;

            return new ComponentValues(this.ComponentCount, c0, c1, c2, c3);
        }
    }
  
    public static void ConvertFromRgbScalar(in ComponentValues values, Span<float> rLane, Span<float> gLane, Span<float> bLane)
    {
        Span<float> y = values.Component0;
        Span<float> cb = values.Component1;
        Span<float> cr = values.Component2;

        for (int i = 0; i < y.Length; i++)
        {
            float r = rLane[i];
            float g = gLane[i];
            float b = bLane[i];

            // y  =   0 + (0.299 * r) + (0.587 * g) + (0.114 * b)
            // cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
            // cr = 128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
            y[i] = (0.299f * r) + (0.587f * g) + (0.114f * b);
            cb[i] = HalfValue - (0.168736f * r) - (0.331264f * g) + (0.5f * b);
            cr[i] = HalfValue + (0.5f * r) - (0.418688f * g) - (0.081312f * b);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplyAdd(
        Vector256<float> va,
        Vector256<float> vm0,
        Vector256<float> vm1)
    {
        if (Fma.IsSupported)
        {
            return Fma.MultiplyAdd(vm1, vm0, va);
        }

        return Avx.Add(Avx.Multiply(vm0, vm1), va);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> MultiplyAdd(
        Vector512<float> va,
        Vector512<float> vm0,
        Vector512<float> vm1)
    {
        if (Avx512F.IsSupported)
        {
            return Avx512F.FusedMultiplyAdd(vm1, vm0, va);
        }

        return (vm0 * vm1) + va;
    }

    public static void ConvertFromRgb256Aligned(in ComponentValues values, Span<float> rLane, Span<float> gLane, Span<float> bLane)
    {
        ref float C0 = ref MemoryMarshal.GetReference(values.Component0);
        ref float C1 = ref MemoryMarshal.GetReference(values.Component1);
        ref float C2 = ref MemoryMarshal.GetReference(values.Component2);

        ref float sr = ref MemoryMarshal.GetReference(rLane);
        ref float sg = ref MemoryMarshal.GetReference(gLane);
        ref float sb = ref MemoryMarshal.GetReference(bLane);

        unsafe
        {
            fixed (float *pC0 = &C0, pC1 = &C1, pC2 = &C2, psr = &sr, psg = &sg, psb = &sb )
            {
                float *ppC0 = (float*)(((nuint)pC0 + (nuint)Vector256<byte>.Count) & ~(nuint)(Vector256<byte>.Count - 1));
                float *ppC1 = (float*)(((nuint)pC0 + (nuint)Vector256<byte>.Count) & ~(nuint)(Vector256<byte>.Count - 1));
                float *ppC2 = (float*)(((nuint)pC0 + (nuint)Vector256<byte>.Count) & ~(nuint)(Vector256<byte>.Count - 1));

                float *ppsr = (float*)(((nuint)psr + (nuint)Vector256<byte>.Count) & ~(nuint)(Vector256<byte>.Count - 1));
                float *ppsg = (float*)(((nuint)psg + (nuint)Vector256<byte>.Count) & ~(nuint)(Vector256<byte>.Count - 1));
                float *ppsb = (float*)(((nuint)psb + (nuint)Vector256<byte>.Count) & ~(nuint)(Vector256<byte>.Count - 1));
            
                // Used for the color conversion
                var chromaOffset = Vector256.Create(HalfValue);

                var f0299 = Vector256.Create(0.299f);
                var f0587 = Vector256.Create(0.587f);
                var f0114 = Vector256.Create(0.114f);
                var fn0168736 = Vector256.Create(-0.168736f);
                var fn0331264 = Vector256.Create(-0.331264f);
                var fn0418688 = Vector256.Create(-0.418688f);
                var fn0081312F = Vector256.Create(-0.081312F);
                var f05 = Vector256.Create(0.5f);

                nuint n = (uint)values.Component0.Length / (uint)Vector256<float>.Count;
                for (nuint i = 0; i < n; i++)
                {
                    nuint offset = (i * (nuint)Vector256<float>.Count);

                    Vector256<float> r = Vector256.LoadAligned(ppsr + offset);
                    Vector256<float> g = Vector256.LoadAligned(ppsg + offset);
                    Vector256<float> b = Vector256.LoadAligned(ppsb + offset);

                    // y  =   0 + (0.299 * r) + (0.587 * g) + (0.114 * b)
                    // cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
                    // cr = 128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
                    Vector256<float> y = MultiplyAdd(MultiplyAdd(Avx.Multiply(f0114, b), f0587, g), f0299, r);
                    Vector256<float> cb = Avx.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx.Multiply(f05, b), fn0331264, g), fn0168736, r));
                    Vector256<float> cr = Avx.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx.Multiply(fn0081312F, b), fn0418688, g), f05, r));

                    y.StoreAligned(ppC0 + offset);
                    cb.StoreAligned(ppC1 + offset);
                    cr.StoreAligned(ppC2 + offset);
                }
            }
        }
    }

    public static void ConvertFromRgb256(in ComponentValues values, Span<float> rLane, Span<float> gLane, Span<float> bLane)
    {
        ref Vector256<float> destY =
            ref Unsafe.As<float, Vector256<float>>(ref MemoryMarshal.GetReference(values.Component0));
        ref Vector256<float> destCb =
            ref Unsafe.As<float, Vector256<float>>(ref MemoryMarshal.GetReference(values.Component1));
        ref Vector256<float> destCr =
            ref Unsafe.As<float, Vector256<float>>(ref MemoryMarshal.GetReference(values.Component2));

        ref Vector256<float> srcR =
            ref Unsafe.As<float, Vector256<float>>(ref MemoryMarshal.GetReference(rLane));
        ref Vector256<float> srcG =
            ref Unsafe.As<float, Vector256<float>>(ref MemoryMarshal.GetReference(gLane));
        ref Vector256<float> srcB =
            ref Unsafe.As<float, Vector256<float>>(ref MemoryMarshal.GetReference(bLane));

        // Used for the color conversion
        var chromaOffset = Vector256.Create(HalfValue);

        var f0299 = Vector256.Create(0.299f);
        var f0587 = Vector256.Create(0.587f);
        var f0114 = Vector256.Create(0.114f);
        var fn0168736 = Vector256.Create(-0.168736f);
        var fn0331264 = Vector256.Create(-0.331264f);
        var fn0418688 = Vector256.Create(-0.418688f);
        var fn0081312F = Vector256.Create(-0.081312F);
        var f05 = Vector256.Create(0.5f);

        nuint n = (uint)values.Component0.Length / (uint)Vector256<float>.Count;
        for (nuint i = 0; i < n; i++)
        {
            Vector256<float> r = Unsafe.Add(ref srcR, i);
            Vector256<float> g = Unsafe.Add(ref srcG, i);
            Vector256<float> b = Unsafe.Add(ref srcB, i);

            // y  =   0 + (0.299 * r) + (0.587 * g) + (0.114 * b)
            // cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
            // cr = 128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
            Vector256<float> y = MultiplyAdd(MultiplyAdd(Avx.Multiply(f0114, b), f0587, g), f0299, r);
            Vector256<float> cb = Avx.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx.Multiply(f05, b), fn0331264, g), fn0168736, r));
            Vector256<float> cr = Avx.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx.Multiply(fn0081312F, b), fn0418688, g), f05, r));

            Unsafe.Add(ref destY, i) = y;
            Unsafe.Add(ref destCb, i) = cb;
            Unsafe.Add(ref destCr, i) = cr;
        }
    }

    public static void ConvertFromRgb512Aligned(in ComponentValues values, Span<float> rLane, Span<float> gLane, Span<float> bLane)
    {
        ref float C0 = ref MemoryMarshal.GetReference(values.Component0);
        ref float C1 = ref MemoryMarshal.GetReference(values.Component1);
        ref float C2 = ref MemoryMarshal.GetReference(values.Component2);

        ref float sr = ref MemoryMarshal.GetReference(rLane);
        ref float sg = ref MemoryMarshal.GetReference(gLane);
        ref float sb = ref MemoryMarshal.GetReference(bLane);

        unsafe
        {
            fixed (float *pC0 = &C0, pC1 = &C1, pC2 = &C2, psr = &sr, psg = &sg, psb = &sb )
            {
                float *ppC0 = (float*)(((nuint)pC0 + (nuint)Vector512<byte>.Count) & ~(nuint)(Vector512<byte>.Count - 1));
                float *ppC1 = (float*)(((nuint)pC0 + (nuint)Vector512<byte>.Count) & ~(nuint)(Vector512<byte>.Count - 1));
                float *ppC2 = (float*)(((nuint)pC0 + (nuint)Vector512<byte>.Count) & ~(nuint)(Vector512<byte>.Count - 1));

                float *ppsr = (float*)(((nuint)psr + (nuint)Vector512<byte>.Count) & ~(nuint)(Vector512<byte>.Count - 1));
                float *ppsg = (float*)(((nuint)psg + (nuint)Vector512<byte>.Count) & ~(nuint)(Vector512<byte>.Count - 1));
                float *ppsb = (float*)(((nuint)psb + (nuint)Vector512<byte>.Count) & ~(nuint)(Vector512<byte>.Count - 1));
            
                // Used for the color conversion
                var chromaOffset = Vector512.Create(HalfValue);

                nuint n = (uint)values.Component0.Length / (uint)Vector512<float>.Count;
                for (nuint i = 0; i < n; i++)
                {
                    nuint offset = (i * (nuint)Vector512<float>.Count);

                    Vector512<float> r = Vector512.LoadAligned(ppsr + offset);
                    Vector512<float> g = Vector512.LoadAligned(ppsg + offset);
                    Vector512<float> b = Vector512.LoadAligned(ppsb + offset);

                    // y  =   0 + (0.299 * r) + (0.587 * g) + (0.114 * b)
                    // cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
                    // cr = 128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
                    Vector512<float> y = MultiplyAdd(MultiplyAdd(Avx512F.Multiply(Vector512.Create(0.114f), b), Vector512.Create(0.587f), g), Vector512.Create(0.299f), r);
                    Vector512<float> cb = Avx512F.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx512F.Multiply(Vector512.Create(0.5f), b), Vector512.Create(-0.331264f), g), Vector512.Create(-0.168736f), r));
                    Vector512<float> cr = Avx512F.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx512F.Multiply(Vector512.Create(-0.081312F), b), Vector512.Create(-0.418688f), g), Vector512.Create(0.5f), r));

                    y.StoreAligned(ppC0 + offset);
                    cb.StoreAligned(ppC1 + offset);
                    cr.StoreAligned(ppC2 + offset);
                }
            }
        }
    }

    public static void ConvertFromRgb512(in ComponentValues values, Span<float> rLane, Span<float> gLane, Span<float> bLane)
    {
        ref Vector512<float> destY =
            ref Unsafe.As<float, Vector512<float>>(ref MemoryMarshal.GetReference(values.Component0));
        ref Vector512<float> destCb =
            ref Unsafe.As<float, Vector512<float>>(ref MemoryMarshal.GetReference(values.Component1));
        ref Vector512<float> destCr =
            ref Unsafe.As<float, Vector512<float>>(ref MemoryMarshal.GetReference(values.Component2));

        ref Vector512<float> srcR =
            ref Unsafe.As<float, Vector512<float>>(ref MemoryMarshal.GetReference(rLane));
        ref Vector512<float> srcG =
            ref Unsafe.As<float, Vector512<float>>(ref MemoryMarshal.GetReference(gLane));
        ref Vector512<float> srcB =
            ref Unsafe.As<float, Vector512<float>>(ref MemoryMarshal.GetReference(bLane));

        // Used for the color conversion
        var chromaOffset = Vector512.Create(HalfValue);

        var f0299 = Vector512.Create(0.299f);
        var f0587 = Vector512.Create(0.587f);
        var f0114 = Vector512.Create(0.114f);
        var fn0168736 = Vector512.Create(-0.168736f);
        var fn0331264 = Vector512.Create(-0.331264f);
        var fn0418688 = Vector512.Create(-0.418688f);
        var fn0081312F = Vector512.Create(-0.081312F);
        var f05 = Vector512.Create(0.5f);

        nuint n = (uint)values.Component0.Length / (uint)Vector512<float>.Count;
        for (nuint i = 0; i < n; i++)
        {
            Vector512<float> r = Unsafe.Add(ref srcR, i);
            Vector512<float> g = Unsafe.Add(ref srcG, i);
            Vector512<float> b = Unsafe.Add(ref srcB, i);

            // y  =   0 + (0.299 * r) + (0.587 * g) + (0.114 * b)
            // cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)
            // cr = 128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)
            Vector512<float> y = MultiplyAdd(MultiplyAdd(Avx512F.Multiply(f0114, b), f0587, g), f0299, r);
            Vector512<float> cb = Avx512F.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx512F.Multiply(f05, b), fn0331264, g), fn0168736, r));
            Vector512<float> cr = Avx512F.Add(chromaOffset, MultiplyAdd(MultiplyAdd(Avx512F.Multiply(fn0081312F, b), fn0418688, g), f05, r));

            Unsafe.Add(ref destY, i) = y;
            Unsafe.Add(ref destCb, i) = cb;
            Unsafe.Add(ref destCr, i) = cr;
        }
    }

    private readonly int componentCount = 3;

    //[Params(100, 1000, 10000, 100000)]
    [Params(100, 1000, 10000)]
    public int Count;

    public const int AlignPad = 16;

    protected Buffer2D<float>[] Input { get; private set; }
    protected float[] rLane;
    protected float[] gLane;
    protected float[] bLane;

    [GlobalSetup]
    public void Setup()
    {
        this.Input = CreateRandomValues(this.componentCount + AlignPad , Count);
        int length = this.Input[0].DangerousGetRowSpan(0).Length;
        this.rLane = new float[length];
        this.gLane = new float[length];
        this.bLane = new float[length];
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Buffer2D<float> buffer in this.Input)
        {
            buffer.Dispose();
        }
    }

    private static Buffer2D<float>[] CreateRandomValues(
        int componentCount,
        int inputBufferLength,
        float minVal = 0f,
        float maxVal = 255f)
    {
        var rnd = new Random(42);
        var buffers = new Buffer2D<float>[componentCount];
        for (int i = 0; i < componentCount; i++)
        {
            var values = new float[inputBufferLength];

            for (int j = 0; j < inputBufferLength; j++)
            {
                values[j] = ((float)rnd.NextDouble() * (maxVal - minVal)) + minVal;
            }

            // no need to dispose when buffer is not array owner
            buffers[i] = Configuration.Default.MemoryAllocator.Allocate2D<float>(values.Length, 1);
        }

        return buffers;
    }

    [Benchmark]
    public void ColorConvertScalar()
    {
        ComponentValues values = new ComponentValues(this.Input, 0);
        ConvertFromRgbScalar(values, rLane, gLane, bLane);
    }

    [Benchmark]
    public void ColorConvertAvx()
    {
        ComponentValues values = new ComponentValues(this.Input, 0);
        ConvertFromRgb256(values, rLane, gLane, bLane);
    }

    [Benchmark]
    public void ColorConvertAvxAligned()
    {
        ComponentValues values = new ComponentValues(this.Input, 0);
        ConvertFromRgb256Aligned(values, rLane, gLane, bLane);
    }

    [Benchmark]
    public void ColorConvertAvx512()
    {
        ComponentValues values = new ComponentValues(this.Input, 0);
        ConvertFromRgb512(values, rLane, gLane, bLane);
    }

    [Benchmark]
    public void ColorConvertAvx512Aligned()
    {
        ComponentValues values = new ComponentValues(this.Input, 0);
        ConvertFromRgb512Aligned(values, rLane, gLane, bLane);
    }

  }
}