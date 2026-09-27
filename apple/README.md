# Logitly on Apple platforms

This directory contains the Swift `Logitly` package and the `LogitlyDemo`
iPhone app. The runtime performs one llama.cpp prefill, requests logits only at
the assistant answer boundary, selects fixed label logits, and applies softmax.
It has no sampling or generation path.

## Mac setup

Requirements: the latest Xcode, macOS 15 or newer, and at least 3 GB of free
space. From the repository root:

```bash
./apple/scripts/bootstrap.sh
open apple/LogitlyDemo.xcodeproj
```

The bootstrap script downloads the pinned 695.8 MB LFM QAD Q4 GGUF, verifies
its exact byte count and SHA-256, installs a pinned local XcodeGen, and creates
the ignored Xcode project. The model is copied into the app bundle by Xcode but
is never tracked by Git.

Select a physical iPhone. Real model loading intentionally fails in the
simulator because the demo requires the Metal backend. Set your development
team in Xcode, build, and run. After the bundled app is installed it performs
no network requests.

## Contract tests

```bash
./apple/scripts/test-contract.sh
```

The current Linux development environment has no Swift or Apple SDK, so the
package and physical-device runtime must be compiled and validated after the
repository is transferred to the Mac.

## Important memory detail

Apple devices use unified memory. A Metal-loaded 696 MB model is visible in the
app's system-memory footprint; that is not a second CPU copy. The benchmark and
500-decision soak test are intended to distinguish this expected residency
from unbounded allocation growth.
