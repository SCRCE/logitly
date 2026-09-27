import Darwin
import Logitly
import SwiftUI

struct BenchmarkResult: Identifiable, Codable {
    let id: Int
    let decisionsPerSecond: Double
    let modelP50Milliseconds: Double
    let modelP95Milliseconds: Double
    let endToEndP50Milliseconds: Double
    let endToEndP95Milliseconds: Double
    let peakBytes: UInt64
    let thermal: String
}

struct BenchmarkView: View {
    @EnvironmentObject private var application: AppModel
    @Environment(\.scenePhase) private var scenePhase
    @State private var results: [BenchmarkResult] = []
    @State private var benchmarkTask: Task<Void, Never>?
    @State private var progress = "Idle"
    @State private var soakResult = "Not run"

    var body: some View {
        NavigationStack {
            List {
                Section { StatusBanner() }
                Section("Protocol") {
                    Text("5 warmup batches · 100 measured decisions · batch sizes 1, 2, 4, 8 · explicit llama.cpp synchronization · zero generated tokens")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(results.isEmpty ? "Run benchmark" : "Run again") {
                        benchmarkTask?.cancel()
                        benchmarkTask = Task { await run() }
                    }
                    Text(progress).font(.caption.monospaced())
                    Button("Run 500-decision memory soak") {
                        benchmarkTask?.cancel()
                        benchmarkTask = Task { await runSoak() }
                    }
                    Text(soakResult).font(.caption.monospaced())
                    if !results.isEmpty {
                        ShareLink(item: exportedResults, subject: Text("Logitly iPhone benchmark")) {
                            Label("Export JSON", systemImage: "square.and.arrow.up")
                        }
                    }
                }
                Section("Results") {
                    ForEach(results) { result in
                        VStack(alignment: .leading, spacing: 7) {
                            HStack {
                                Text("Batch \(result.id)").font(.headline)
                                Spacer()
                                Text("\(result.decisionsPerSecond, specifier: "%.1f") decisions/s").fontWeight(.semibold)
                            }
                            Text("model p50/p95: \(result.modelP50Milliseconds, specifier: "%.2f") / \(result.modelP95Milliseconds, specifier: "%.2f") ms per decision")
                            Text("end-to-end p50/p95: \(result.endToEndP50Milliseconds, specifier: "%.2f") / \(result.endToEndP95Milliseconds, specifier: "%.2f") ms per decision")
                            Text("peak footprint: \(ByteCountFormatter.string(fromByteCount: Int64(result.peakBytes), countStyle: .memory)) · thermal: \(result.thermal)")
                        }
                        .font(.caption.monospacedDigit())
                    }
                }
            }
            .navigationTitle("Benchmark")
            .onChange(of: scenePhase) { _, phase in
                if phase != .active {
                    benchmarkTask?.cancel()
                    progress = "Paused while app is inactive"
                }
            }
        }
    }

    @MainActor
    private func run() async {
        results = []
        do {
            let model = try await application.ensureLoaded()
            for batchSize in [1, 2, 4, 8] {
                try Task.checkCancellation()
                progress = "Warming batch \(batchSize)…"
                let requests = makeRequests(count: batchSize)
                for _ in 0..<5 { _ = try await model.decideMany(requests, batchSize: batchSize) }
                var modelTimes: [Double] = []
                var endTimes: [Double] = []
                var peak = AppMemory.physicalFootprint()
                let benchmarkStart = ContinuousClock.now
                var completed = 0
                while completed < 100 {
                    try Task.checkCancellation()
                    let count = min(batchSize, 100 - completed)
                    let current = Array(requests.prefix(count))
                    let started = ContinuousClock.now
                    _ = try await model.decideMany(current, batchSize: count)
                    let elapsed = started.duration(to: .now).seconds / Double(count)
                    let diagnostics = await model.diagnostics
                    modelTimes.append(diagnostics.modelOnlySeconds / Double(count))
                    endTimes.append(elapsed)
                    completed += count
                    peak = max(peak, AppMemory.physicalFootprint())
                    progress = "Batch \(batchSize): \(completed)/100"
                    await Task.yield()
                }
                let total = benchmarkStart.duration(to: .now).seconds
                results.append(BenchmarkResult(
                    id: batchSize,
                    decisionsPerSecond: 100 / total,
                    modelP50Milliseconds: percentile(modelTimes, 0.50) * 1_000,
                    modelP95Milliseconds: percentile(modelTimes, 0.95) * 1_000,
                    endToEndP50Milliseconds: percentile(endTimes, 0.50) * 1_000,
                    endToEndP95Milliseconds: percentile(endTimes, 0.95) * 1_000,
                    peakBytes: peak,
                    thermal: ProcessInfo.processInfo.thermalState.label
                ))
            }
            progress = "Complete"
        } catch is CancellationError {
            progress = "Cancelled"
        } catch {
            application.lastError = error.localizedDescription
            progress = "Failed: \(error.localizedDescription)"
        }
    }

    private func makeRequests(count: Int) -> [DecisionRequest] {
        (0..<count).map { index in
            .choice(ChoiceRequest(
                id: "benchmark-\(index)",
                state: .json(["priority": .string("high"), "attempt": .integer(Int64(index))]),
                question: "What action should be taken?",
                choices: [
                    ChoiceOption(id: "approve", text: "Approve the request"),
                    ChoiceOption(id: "review", text: "Request manual review"),
                    ChoiceOption(id: "reject", text: "Reject the request"),
                ]
            ))
        }
    }

    @MainActor
    private func runSoak() async {
        do {
            let model = try await application.ensureLoaded()
            let request = makeRequests(count: 1)
            for _ in 0..<5 { _ = try await model.decideMany(request) }
            let start = AppMemory.physicalFootprint()
            var peak = start
            for index in 1...500 {
                try Task.checkCancellation()
                _ = try await model.decideMany(request)
                peak = max(peak, AppMemory.physicalFootprint())
                if index.isMultiple(of: 10) {
                    progress = "Memory soak: \(index)/500"
                    await Task.yield()
                }
            }
            let end = AppMemory.physicalFootprint()
            let delta = Int64(end) - Int64(start)
            soakResult = "start \(ByteCountFormatter.string(fromByteCount: Int64(start), countStyle: .memory)); end \(ByteCountFormatter.string(fromByteCount: Int64(end), countStyle: .memory)); Δ \(ByteCountFormatter.string(fromByteCount: delta, countStyle: .memory)); peak \(ByteCountFormatter.string(fromByteCount: Int64(peak), countStyle: .memory))"
            progress = "Soak complete"
        } catch is CancellationError {
            progress = "Cancelled"
        } catch {
            application.lastError = error.localizedDescription
            progress = "Failed: \(error.localizedDescription)"
        }
    }

    private var exportedResults: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? String(decoding: encoder.encode(results), as: UTF8.self)) ?? "[]"
    }

    private func percentile(_ values: [Double], _ fraction: Double) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        return sorted[min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1)]
    }
}

enum AppMemory {
    static func physicalFootprint() -> UInt64 {
        var information = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &information) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? UInt64(information.phys_footprint) : 0
    }
}

private extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}

private extension Duration {
    var seconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
