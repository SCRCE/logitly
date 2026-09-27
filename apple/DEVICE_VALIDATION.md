# Physical-device validation

Run these checks on the destination Mac and iPhone before calling the Apple
runtime complete.

1. Run `./apple/scripts/bootstrap.sh` from the repository root.
2. Run `./apple/scripts/test-macos.sh` and resolve any Xcode/SDK compatibility
   issue before signing the app.
3. Open `apple/LogitlyDemo.xcodeproj`, select the `LogitlyDemo` target, choose
   the local development team, and select a physical iPhone running iOS 18 or
   newer.
4. Build and launch once. The About screen must show Metal, one prefill, and
   zero generated tokens. Model loading must fail rather than use a CPU-only
   backend when Metal is unavailable.
5. Exercise Choice, Noul, and Score in the Playground. Inspect the rendered
   answer boundary and the one-token label IDs.
6. Run the Benchmark screen. Export the JSON for batch sizes 1, 2, 4, and 8.
   Record any batch that fails to fit instead of reducing it silently.
7. Run the 500-decision memory soak. After five warmups, the final footprint
   must grow by no more than 25 MB and batch-1 peak must remain below 2 GB.
8. Reset Snake, run 50 steps, and export the trace. Every record must contain
   the raw and executed moves, probability distribution, shield flag, food
   event, cumulative food count, and end-to-end time.
9. Relaunch in airplane mode and repeat a decision to prove runtime inference
   has no network dependency.
10. Archive with Xcode's Release configuration, validate the archive, and send
    it to the selected internal TestFlight group. Confirm the LFM and llama.cpp
    license resources are visible before distribution.

Apple uses unified memory, so the model mapping is expected to appear in the
app's physical footprint even while Metal evaluates it. The acceptance check
is bounded residency and a flat repeated-decision footprint, not zero system
memory use.
