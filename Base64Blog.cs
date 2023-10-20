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
  public class Base64Tests
  {
      [Params(10, 100, 1000, 10000)]
      public int NumberOfBytes { get; set; }

      private byte[] _decodedBytes;
      private byte[] _encodedBytes;

      private char[] _encodedChars;

      [GlobalSetup(Target = nameof(Base64Encode))]
      public void SetupBase64Encode()
      {
          _decodedBytes = ValuesGenerator.Array<byte>(NumberOfBytes);
          _encodedBytes = new byte[Base64.GetMaxEncodedToUtf8Length(NumberOfBytes)];
      }

      [Benchmark]
      public OperationStatus Base64Encode() => Base64.EncodeToUtf8(_decodedBytes, _encodedBytes, out _, out _);

      [GlobalSetup(Target = nameof(Base64EncodeDestinationTooSmall))]
      public void SetupBase64EncodeDestinationTooSmall()
      {
          _decodedBytes = ValuesGenerator.Array<byte>(NumberOfBytes);
          _encodedBytes = new byte[Base64.GetMaxEncodedToUtf8Length(NumberOfBytes) - 1]; // -1
      }

      [Benchmark]
      public OperationStatus Base64EncodeDestinationTooSmall() => Base64.EncodeToUtf8(_decodedBytes, _encodedBytes, out _, out _);

      [GlobalSetup(Target = nameof(ConvertToBase64CharArray))]
      public void SetupConvertToBase64CharArray()
      {
          _decodedBytes = ValuesGenerator.Array<byte>(NumberOfBytes);
          _encodedChars = new char[Base64.GetMaxEncodedToUtf8Length(NumberOfBytes)];
      }

      [Benchmark]
      public int ConvertToBase64CharArray() => Convert.ToBase64CharArray(_decodedBytes, 0, _decodedBytes.Length, _encodedChars, 0);

      [GlobalSetup(Target = nameof(Base64Decode))]
      public void SetupBase64Decode()
      {
          _encodedBytes = ValuesGenerator.ArrayBase64EncodingBytes(NumberOfBytes);
          _decodedBytes = new byte[Base64.GetMaxEncodedToUtf8Length(NumberOfBytes)];
      }

      [Benchmark]
      public OperationStatus Base64Decode() => Base64.DecodeFromUtf8(_encodedBytes, _decodedBytes, out _, out _);

      [GlobalSetup(Target = nameof(Base64DecodeDestinationTooSmall))]
      public void SetupBase64DecodeDestinationTooSmall()
      {
          _encodedBytes = ValuesGenerator.ArrayBase64EncodingBytes(NumberOfBytes);
          _decodedBytes = new byte[Base64.GetMaxEncodedToUtf8Length(NumberOfBytes) - 1];
      }

      [Benchmark]
      public OperationStatus Base64DecodeDestinationTooSmall() => Base64.DecodeFromUtf8(_encodedBytes, _decodedBytes, out _, out _);

#if !NETFRAMEWORK // API added in .NET Core 2.1
      [GlobalSetup(Target = nameof(ConvertTryFromBase64Chars))]
      public void SetupConvertTryFromBase64Chars()
      {
          _decodedBytes = ValuesGenerator.Array<byte>(NumberOfBytes);
          _encodedChars = Convert.ToBase64String(_decodedBytes).ToCharArray();
      }

      [Benchmark]
      public bool ConvertTryFromBase64Chars() => Convert.TryFromBase64Chars(_encodedChars, _decodedBytes, out _);
#endif
  }

  // We want to test InPlace methods, which require fresh input for every benchmark invocation.
  // To setup every benchmark invocation we are using [IterationSetup].
  // To make the results stable the Iteration needs to last at least 100ms, this is why we are using bigger value for NumberOfBytes
  // Due to limitation of BDN, where Params have no Target and are applied to entire class the benchmarks live in a separate class.
  [WarmupCount(30)] // make sure it's promoted to Tier 1
  public class Base64EncodeDecodeInPlaceTests
  {
      [Params(1000 * 1000 * 200)] // allows for stable iteration around 200ms
      public int NumberOfBytes { get; set; }

      private byte[] _source;
      private byte[] _destination;

      [GlobalSetup]
      public void Setup()
      {
          _source = ValuesGenerator.Array<byte>(NumberOfBytes);
          _destination = new byte[Base64.GetMaxEncodedToUtf8Length(NumberOfBytes)];
      }

      [IterationSetup(Target = nameof(Base64EncodeInPlace))]
      public void SetupBase64EncodeInPlace() => Array.Copy(_source, _destination, _source.Length);

      [Benchmark]
      public OperationStatus Base64EncodeInPlace() => Base64.EncodeToUtf8InPlace(_destination, _source.Length, out _);

      [IterationSetup(Target = nameof(Base64DecodeInPlace))]
      public void SetupBase64DecodeInPlace() => Base64.EncodeToUtf8(_source, _destination, out _, out _);

      [Benchmark]
      public OperationStatus Base64DecodeInPlace() => Base64.DecodeFromUtf8InPlace(_destination, out _);
  }
}