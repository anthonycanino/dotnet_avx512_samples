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
using BenchmarkDotNet.Toolchains.CoreRun;
using BenchmarkDotNet.Jobs;

namespace Avx512Blog
{
  public class Program
  {
    static string corerun_path = "C:\\Users\\acanino\\Dev\\dotnet\\runtime\\artifacts\\bin\\testhost\\net9.0-windows-Release-x64\\shared\\Microsoft.NETCore.App\\9.0.0\\corerun.exe";
    
    static void Main(string[] args)
    {
      var toolchain = new CoreRunToolchain(new FileInfo(corerun_path), targetFrameworkMoniker: "net8.0");

      var config = DefaultConfig.Instance
        .AddJob(Job.Default.WithToolchain(toolchain).WithEnvironmentVariables(new EnvironmentVariable("DOTNET_EnableAVX512F", "0")).WithId("Vector256"))
        .AddJob(Job.Default.WithToolchain(toolchain).WithId("Vector512"));

      var switcher = new BenchmarkSwitcher(new[] {
        typeof(TensorBlog),
        typeof(Base64Tests),
        typeof(Base64EncodeDecodeInPlaceTests),
        typeof(MemEqualTests<>),
        typeof(TensorBlog),
        typeof(ColorConversionBlog),
        typeof(Crc32Tests)
      });
      switcher.Run(args, config);
    }
  }
}