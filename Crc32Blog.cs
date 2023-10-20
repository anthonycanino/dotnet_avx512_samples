
using BenchmarkDotNet.Attributes;
using System.IO.Hashing;

namespace Avx512Blog
{
    public class Crc32Tests
    {
        [Params(
            512, 
            1000,
            1024,
            10000
            )] // we are only testing on the vector path, 512 bytes are the minimal input size.
        public int Size {get; set;}


        private byte[] src;

        [GlobalSetup(Target = nameof(CRCTest))]
        public void SetupCRC()
        {
            src = ValuesGenerator.Array<byte>(Size);
        }

        [Benchmark]
        public byte[] CRCTest() => Crc32.Hash(new Span<byte>(src));

    }
}