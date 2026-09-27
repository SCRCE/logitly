import Foundation

public actor DecisionModel {
    private var runtime: (any DirectLogitRuntime)?
    private let profile: ModelProfile
    private let configuration: ModelConfiguration
    public private(set) var diagnostics: InferenceDiagnostics
    public private(set) var lastInspection: PromptInspection?

    private init(runtime: any DirectLogitRuntime, profile: ModelProfile, configuration: ModelConfiguration) {
        self.runtime = runtime
        self.profile = profile
        self.configuration = configuration
        diagnostics = InferenceDiagnostics(backend: runtime.backendName, modelOnlySeconds: 0, generatedTokens: 0)
        lastInspection = nil
    }

    public static func load(
        modelURL: URL,
        profile: ModelProfile = .generic,
        configuration: ModelConfiguration = .iphoneDefault
    ) async throws -> DecisionModel {
        let runtime = try await Task.detached(priority: .userInitiated) {
            try NativeLlamaRuntime(modelURL: modelURL, profile: profile, configuration: configuration)
        }.value
        return DecisionModel(runtime: runtime, profile: profile, configuration: configuration)
    }

    init(testRuntime: any DirectLogitRuntime, profile: ModelProfile = .generic, configuration: ModelConfiguration = .iphoneDefault) {
        runtime = testRuntime
        self.profile = profile
        self.configuration = configuration
        diagnostics = InferenceDiagnostics(backend: testRuntime.backendName, modelOnlySeconds: 0, generatedTokens: 0)
        lastInspection = nil
    }

    public func noul(state: DecisionState, question: String) throws -> NoulResult {
        let results = try decideMany([.noul(NoulRequest(state: state, question: question))])
        guard case let .noul(_, result) = results[0] else { preconditionFailure("invalid result type") }
        return result
    }

    public func choice(state: DecisionState, question: String, choices: [ChoiceOption]) throws -> ChoiceResult {
        let results = try decideMany([.choice(ChoiceRequest(state: state, question: question, choices: choices))])
        guard case let .choice(_, result) = results[0] else { preconditionFailure("invalid result type") }
        return result
    }

    public func score(state: DecisionState, question: String, levels: [String]) throws -> ScoreResult {
        let results = try decideMany([.score(ScoreRequest(state: state, question: question, levels: levels))])
        guard case let .score(_, result) = results[0] else { preconditionFailure("invalid result type") }
        return result
    }

    public func decideMany(_ requests: [DecisionRequest], batchSize: Int? = nil) throws -> [DecisionResult] {
        guard let runtime else { throw LogitlyError.modelUnloaded }
        guard !requests.isEmpty else { return [] }
        let requestedBatch = batchSize ?? 1
        guard requestedBatch > 0, requestedBatch <= configuration.maxBatchSize else {
            throw LogitlyError.invalidRequest("batchSize must be between 1 and \(configuration.maxBatchSize)")
        }
        let prepared = try requests.map(prepare)
        var results: [DecisionResult] = []
        for start in stride(from: 0, to: prepared.count, by: requestedBatch) {
            let end = min(start + requestedBatch, prepared.count)
            let group = Array(prepared[start..<end])
            let output = try runtime.score(prompts: group.map(\.prompt), labels: group.map(\.labels))
            guard output.logits.count == group.count,
                  output.rendered.count == group.count,
                  output.promptTokens.count == group.count,
                  output.labelTokenIDs.count == group.count else {
                throw LogitlyError.logitsUnavailable
            }
            diagnostics = InferenceDiagnostics(
                backend: runtime.backendName,
                modelOnlySeconds: output.modelOnlySeconds,
                generatedTokens: 0
            )
            if let last = group.indices.last {
                lastInspection = PromptInspection(
                    prompt: group[last].prompt,
                    renderedPrompt: output.rendered[last],
                    promptTokenIDs: output.promptTokens[last],
                    labelTokenIDs: output.labelTokenIDs[last]
                )
            }
            for (decision, logits) in zip(group, output.logits) {
                results.append(try result(for: decision, probabilities: RestrictedSoftmax.probabilities(logits)))
            }
        }
        return results
    }

    public func inspect(_ request: DecisionRequest) throws -> PromptInspection {
        guard let runtime else { throw LogitlyError.modelUnloaded }
        let prepared = try prepare(request)
        let output = try runtime.score(prompts: [prepared.prompt], labels: [prepared.labels])
        return PromptInspection(
            prompt: prepared.prompt,
            renderedPrompt: output.rendered[0],
            promptTokenIDs: output.promptTokens[0],
            labelTokenIDs: output.labelTokenIDs[0]
        )
    }

    public func unload() {
        runtime?.unload()
        runtime = nil
        lastInspection = nil
    }

    private func prepare(_ request: DecisionRequest) throws -> PreparedDecision {
        switch request {
        case let .noul(request):
            let id = try StateSerializer.text(request.id, name: "request id", maximum: ContractLimits.identifierCharacters)
            let state = try StateSerializer.serialize(request.state)
            let question = try StateSerializer.text(request.question, name: "question", maximum: ContractLimits.questionCharacters)
            return PreparedDecision(
                id: id,
                kind: .noul,
                prompt: PromptRenderer.noul(state: state, question: question),
                labels: profile.noulLabels,
                semanticIDs: ["yes", "no"]
            )
        case let .choice(request):
            let id = try StateSerializer.text(request.id, name: "request id", maximum: ContractLimits.identifierCharacters)
            let state = try StateSerializer.serialize(request.state)
            let question = try StateSerializer.text(request.question, name: "question", maximum: ContractLimits.questionCharacters)
            guard (2...20).contains(request.choices.count) else {
                throw LogitlyError.invalidRequest("choice requires between 2 and 20 options")
            }
            let ids = try request.choices.map { try StateSerializer.text($0.id, name: "choice id", maximum: ContractLimits.identifierCharacters) }
            guard Set(ids).count == ids.count else { throw LogitlyError.invalidRequest("choice ids must be unique") }
            let descriptions = try request.choices.map { try StateSerializer.text($0.text, name: "choice", maximum: ContractLimits.criterionCharacters) }
            let labels = Array(profile.choiceLabels.prefix(ids.count))
            guard labels.count == ids.count else { throw LogitlyError.invalidRequest("model profile has too few choice labels") }
            return PreparedDecision(
                id: id,
                kind: .choice,
                prompt: PromptRenderer.choice(state: state, question: question, labels: labels, choices: descriptions),
                labels: labels,
                semanticIDs: ids
            )
        case let .score(request):
            let id = try StateSerializer.text(request.id, name: "request id", maximum: ContractLimits.identifierCharacters)
            let state = try StateSerializer.serialize(request.state)
            let question = try StateSerializer.text(request.question, name: "question", maximum: ContractLimits.questionCharacters)
            guard (2...10).contains(request.levels.count) else {
                throw LogitlyError.invalidRequest("score requires between 2 and 10 levels")
            }
            let levels = try request.levels.map { try StateSerializer.text($0, name: "level", maximum: ContractLimits.criterionCharacters) }
            guard Set(levels).count == levels.count else { throw LogitlyError.invalidRequest("score levels must be unique") }
            let labels = Array(profile.scoreLabels.prefix(levels.count))
            guard labels.count == levels.count else { throw LogitlyError.invalidRequest("model profile has too few score labels") }
            return PreparedDecision(
                id: id,
                kind: .score,
                prompt: PromptRenderer.score(state: state, question: question, labels: labels, levels: levels),
                labels: labels,
                semanticIDs: levels.indices.map(String.init)
            )
        }
    }

    private func result(for decision: PreparedDecision, probabilities: [Double]) throws -> DecisionResult {
        guard probabilities.count == decision.semanticIDs.count,
              abs(probabilities.reduce(0, +) - 1) <= 1e-6 else {
            throw LogitlyError.logitsUnavailable
        }
        let winner = probabilities.indices.max { probabilities[$0] < probabilities[$1] }!
        switch decision.kind {
        case .noul:
            return .noul(id: decision.id, NoulResult(yes: probabilities[0], no: probabilities[1]))
        case .choice:
            return .choice(
                id: decision.id,
                ChoiceResult(
                    probabilities: Dictionary(uniqueKeysWithValues: zip(decision.semanticIDs, probabilities)),
                    choice: decision.semanticIDs[winner],
                    confidence: probabilities[winner]
                )
            )
        case .score:
            let expected = zip(probabilities.indices, probabilities).reduce(0.0) { $0 + Double($1.0) * $1.1 }
            return .score(
                id: decision.id,
                ScoreResult(distribution: probabilities, score: expected, confidence: probabilities[winner])
            )
        }
    }
}
