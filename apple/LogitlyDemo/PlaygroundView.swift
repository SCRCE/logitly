import Logitly
import SwiftUI

struct PlaygroundView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case choice = "Choice"
        case noul = "Noul"
        case score = "Score"
        var id: Self { self }
    }

    @EnvironmentObject private var application: AppModel
    @State private var mode = Mode.choice
    @State private var state = "{\"balance\":420,\"status\":\"pending\"}"
    @State private var question = "What action should be taken?"
    @State private var options = "approve | Approve\nreview | Request review\nreject | Reject"
    @State private var probabilities: [(String, Double)] = []
    @State private var selected = ""
    @State private var confidence = 0.0
    @State private var expectedScore: Double?
    @State private var renderedPrompt = ""
    @State private var tokenSummary = ""
    @State private var running = false

    var body: some View {
        NavigationStack {
            Form {
                Section { StatusBanner() }
                Section("Decision") {
                    Picker("Primitive", selection: $mode) {
                        ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    TextField("State", text: $state, axis: .vertical).lineLimit(3...8)
                    TextField("Question", text: $question, axis: .vertical).lineLimit(2...5)
                    if mode != .noul {
                        TextField(mode == .choice ? "id | description, one per line" : "One level per line", text: $options, axis: .vertical)
                            .lineLimit(3...10)
                    }
                    Button {
                        Task { await run() }
                    } label: {
                        HStack {
                            if running { ProgressView() }
                            Text("Run one prefill")
                        }
                    }
                    .disabled(running)
                }
                if !probabilities.isEmpty {
                    Section("Restricted probabilities") {
                        ForEach(probabilities, id: \.0) { item in
                            ProbabilityRow(label: item.0, value: item.1, selected: item.0 == selected)
                        }
                        LabeledContent("Choice", value: selected)
                        LabeledContent("Confidence", value: confidence.formatted(.percent.precision(.fractionLength(2))))
                        if let expectedScore { LabeledContent("Expected score", value: expectedScore.formatted(.number.precision(.fractionLength(3)))) }
                    }
                    Section("Answer boundary") {
                        Text(renderedPrompt).font(.caption.monospaced()).textSelection(.enabled)
                        Text(tokenSummary).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
                if let error = application.lastError {
                    Section("Error") { Text(error).foregroundStyle(.red) }
                }
            }
            .navigationTitle("Logitly")
        }
    }

    @MainActor
    private func run() async {
        running = true
        defer { running = false }
        do {
            let model = try await application.ensureLoaded()
            let decisionState: DecisionState
            if let data = state.data(using: .utf8), let json = try? JSONDecoder().decode(JSONValue.self, from: data) {
                decisionState = .json(json)
            } else {
                decisionState = .text(state)
            }
            expectedScore = nil
            switch mode {
            case .choice:
                let choices = options.split(separator: "\n").enumerated().map { index, line in
                    let fields = line.split(separator: "|", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
                    return ChoiceOption(id: fields.count == 2 ? fields[0] : "choice-\(index + 1)", text: fields.last ?? String(line))
                }
                let result = try await model.choice(state: decisionState, question: question, choices: choices)
                probabilities = choices.map { ($0.id, result.probabilities[$0.id] ?? 0) }
                selected = result.choice
                confidence = result.confidence
            case .noul:
                let result = try await model.noul(state: decisionState, question: question)
                probabilities = [("yes", result.yes), ("no", result.no)]
                selected = result.yes >= result.no ? "yes" : "no"
                confidence = max(result.yes, result.no)
            case .score:
                let levels = options.split(separator: "\n").map(String.init)
                let result = try await model.score(state: decisionState, question: question, levels: levels)
                probabilities = zip(levels, result.distribution).map { ($0, $1) }
                let winner = result.distribution.indices.max { result.distribution[$0] < result.distribution[$1] } ?? 0
                selected = levels[winner]
                confidence = result.confidence
                expectedScore = result.score
            }
            if let inspection = await model.lastInspection {
                renderedPrompt = inspection.renderedPrompt
                tokenSummary = "prompt tokens: \(inspection.promptTokenIDs.count)\nlabels: " + inspection.labelTokenIDs.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: "  ")
            }
            application.lastError = nil
        } catch {
            application.lastError = error.localizedDescription
        }
    }
}
