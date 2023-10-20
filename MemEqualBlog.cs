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

using System.Buffers;
using System.Buffers.Text;

namespace Avx512Blog
{
    [GenericTypeArguments(typeof(byte))]
    [GenericTypeArguments(typeof(char))]
    [GenericTypeArguments(typeof(int))]
    public class MemEqualTests<T>
    {
        [Params(10, 100, 1000, 10000)] 
        public int Size;

        private T[] _array, _same;

        [GlobalSetup]
        public void Setup()
        {
            T[] array = ValuesGenerator.Array<T>(Size * 2);
            _array = array.Take(Size).ToArray();
            _same = _array.ToArray();
        }

        [Benchmark]
        public bool SequenceEqual() => new System.Span<T>(_array).SequenceEqual(new System.ReadOnlySpan<T>(_same));
    }
}