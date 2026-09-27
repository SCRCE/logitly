import Logitly
import SwiftUI

struct SnakeView: View {
    @EnvironmentObject private var application: AppModel
    @State private var game = SnakeGame()
    @State private var records: [SnakeRecord] = []
    @State private var running = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    StatusBanner()
                    VStack(spacing: 3) {
                        ForEach(0..<game.height, id: \.self) { y in
                            HStack(spacing: 3) {
                                ForEach(0..<game.width, id: \.self) { x in
                                    cell(Cell(x: x, y: y))
                                }
                            }
                        }
                    }
                    .padding(12)
                    .background(.black.opacity(0.9), in: RoundedRectangle(cornerRadius: 16))
                    HStack {
                        metric("Steps", "\(game.steps)")
                        metric("Food", "\(game.foodEaten)")
                        metric("Shields", "\(records.filter(\.shielded).count)")
                    }
                    Text("Planner-directed mode: the deterministic navigation planner supplies the recommended direction; LFM extracts that choice from one prefill.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button("One step") { Task { await step() } }.buttonStyle(.bordered)
                        Button("Run to 50") { Task { await runToFifty() } }.buttonStyle(.borderedProminent)
                        Button("Reset") { game = SnakeGame(); records = [] }.buttonStyle(.bordered)
                    }
                    .disabled(running)
                    if !records.isEmpty {
                        ShareLink(item: exportedTrace, subject: Text("Logitly Snake trace")) {
                            Label("Export trace", systemImage: "square.and.arrow.up")
                        }
                    }
                    if let last = records.last {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Latest decision").font(.headline)
                            ForEach(Direction.allCases, id: \.self) { direction in
                                ProbabilityRow(
                                    label: direction.rawValue,
                                    value: last.probabilities[direction.rawValue] ?? 0,
                                    selected: direction == last.executed
                                )
                            }
                            Text("raw=\(last.raw.rawValue) executed=\(last.executed.rawValue) · \(last.seconds.formatted(.number.precision(.fractionLength(4))))s")
                                .font(.caption.monospaced()).foregroundStyle(.secondary)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Snake")
        }
    }

    @ViewBuilder
    private func cell(_ cell: Cell) -> some View {
        let isHead = game.body.first == cell
        let isBody = game.body.contains(cell)
        let isFood = game.food == cell
        RoundedRectangle(cornerRadius: 4)
            .fill(isHead ? Color.mint : isBody ? Color.green : isFood ? Color.orange : Color.white.opacity(0.1))
            .frame(width: 31, height: 31)
            .overlay { if isFood { Image(systemName: "circle.fill").font(.caption).foregroundStyle(.yellow) } }
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack { Text(value).font(.title2.bold()).monospacedDigit(); Text(label).font(.caption).foregroundStyle(.secondary) }
            .frame(maxWidth: .infinity)
    }

    @MainActor
    private func step() async {
        guard game.food != nil else { return }
        running = true
        defer { running = false }
        do {
            let model = try await application.ensureLoaded()
            let request = game.decisionRequest()
            let start = ContinuousClock.now
            let decisions = try await model.decideMany([.choice(request)])
            let seconds = start.duration(to: .now).seconds
            guard case let .choice(_, result) = decisions[0], let raw = Direction(rawValue: result.choice) else { return }
            let outcome = game.move(raw: raw, probabilities: result.probabilities)
            records.append(SnakeRecord(
                id: game.steps,
                raw: raw,
                executed: outcome.executed,
                shielded: outcome.shielded,
                ate: outcome.ate,
                foodEaten: game.foodEaten,
                probabilities: result.probabilities,
                seconds: seconds
            ))
        } catch { application.lastError = error.localizedDescription }
    }

    @MainActor
    private func runToFifty() async {
        running = true
        defer { running = false }
        do {
            let model = try await application.ensureLoaded()
            while game.steps < 50, game.food != nil, !Task.isCancelled {
                let request = game.decisionRequest()
                let start = ContinuousClock.now
                let decisions = try await model.decideMany([.choice(request)])
                let seconds = start.duration(to: .now).seconds
                guard case let .choice(_, result) = decisions[0], let raw = Direction(rawValue: result.choice) else { break }
                let outcome = game.move(raw: raw, probabilities: result.probabilities)
                records.append(SnakeRecord(id: game.steps, raw: raw, executed: outcome.executed, shielded: outcome.shielded, ate: outcome.ate, foodEaten: game.foodEaten, probabilities: result.probabilities, seconds: seconds))
                await Task.yield()
            }
        } catch { application.lastError = error.localizedDescription }
    }

    private var exportedTrace: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? String(decoding: encoder.encode(records), as: UTF8.self)) ?? "[]"
    }
}

private extension Duration {
    var seconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
