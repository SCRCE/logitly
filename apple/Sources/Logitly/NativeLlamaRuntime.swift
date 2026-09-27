import Foundation
import llama

struct RuntimeBatch: Sendable {
    let logits: [[Float]]
    let rendered: [String]
    let promptTokens: [[Int32]]
    let labelTokenIDs: [[String: Int32]]
    let modelOnlySeconds: Double
}

protocol DirectLogitRuntime: AnyObject, Sendable {
    var backendName: String { get }
    func score(prompts: [String], labels: [[String]]) throws -> RuntimeBatch
    func unload()
}

final class NativeLlamaRuntime: DirectLogitRuntime, @unchecked Sendable {
    private static let initializeBackend: Void = { llama_backend_init() }()

    private let profile: ModelProfile
    private let configuration: ModelConfiguration
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var template: UnsafePointer<CChar>?
    private(set) var backendName = "Metal"

    init(modelURL: URL, profile: ModelProfile, configuration: ModelConfiguration) throws {
        _ = Self.initializeBackend
        self.profile = profile
        self.configuration = configuration

        guard profile.choiceLabels.count >= 20,
              profile.noulLabels.count == 2,
              profile.scoreLabels.count >= 10,
              Set(profile.choiceLabels).count == profile.choiceLabels.count,
              Set(profile.noulLabels).count == profile.noulLabels.count,
              Set(profile.scoreLabels).count == profile.scoreLabels.count,
              (profile.choiceLabels + profile.noulLabels + profile.scoreLabels).allSatisfy({ !$0.isEmpty }) else {
            throw LogitlyError.invalidRequest("Model profile labels are malformed or duplicated")
        }
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw LogitlyError.modelNotFound(modelURL.path)
        }
        guard configuration.maxInputTokens > 0,
              (1...8).contains(configuration.maxBatchSize),
              configuration.microBatchTokens > 0 else {
            throw LogitlyError.invalidRequest("Invalid model configuration")
        }

#if targetEnvironment(simulator)
        if configuration.requireMetal { throw LogitlyError.unsupportedSimulator }
#endif
        if configuration.requireMetal {
            guard llama_supports_gpu_offload(), Self.hasMetalDevice() else {
                throw LogitlyError.metalUnavailable
            }
        }

        var modelParameters = llama_model_default_params()
        modelParameters.n_gpu_layers = -1
        modelParameters.split_mode = LLAMA_SPLIT_MODE_NONE
        modelParameters.main_gpu = 0

        model = modelURL.path.withCString { llama_model_load_from_file($0, modelParameters) }
        guard let model else { throw LogitlyError.modelLoadFailed }
        vocab = llama_model_get_vocab(model)
        template = llama_model_chat_template(model, nil)
        guard vocab != nil else { throw LogitlyError.modelLoadFailed }
        guard template != nil else { throw LogitlyError.chatTemplateMissing }

        var contextParameters = llama_context_default_params()
        let totalContext = configuration.maxInputTokens * configuration.maxBatchSize
        contextParameters.n_ctx = UInt32(totalContext)
        contextParameters.n_batch = UInt32(totalContext)
        contextParameters.n_ubatch = UInt32(min(configuration.microBatchTokens, totalContext))
        contextParameters.n_seq_max = UInt32(configuration.maxBatchSize)
        // Only one token per sequence requests logits. Without these limits,
        // llama.cpp may reserve n_batch full-vocabulary output rows.
        contextParameters.n_outputs_max = UInt32(configuration.maxBatchSize)
        contextParameters.n_outputs_max_per_seq = 1
        let automaticThreads = max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 2))
        let threads = configuration.threads > 0 ? configuration.threads : automaticThreads
        contextParameters.n_threads = Int32(threads)
        contextParameters.n_threads_batch = Int32(threads)
        contextParameters.offload_kqv = true
        contextParameters.op_offload = true
        context = llama_init_from_model(model, contextParameters)
        guard context != nil else { throw LogitlyError.contextCreationFailed }

        do {
            try validateAllVerbalizers()
        } catch {
            unload()
            throw error
        }
    }

    deinit { unload() }

    func score(prompts: [String], labels: [[String]]) throws -> RuntimeBatch {
        guard let context else { throw LogitlyError.modelUnloaded }
        guard prompts.count == labels.count, !prompts.isEmpty else {
            throw LogitlyError.invalidRequest("Prompt and label batches must have the same non-zero size")
        }
        guard prompts.count <= configuration.maxBatchSize else {
            throw LogitlyError.invalidRequest("Batch exceeds the configured maximum of \(configuration.maxBatchSize)")
        }

        var rendered: [String] = []
        var tokenRows: [[Int32]] = []
        var labelRows: [[String: Int32]] = []
        for (prompt, rowLabels) in zip(prompts, labels) {
            let answerBoundary = try render(prompt)
            let tokens = try tokenize(answerBoundary)
            guard tokens.count <= configuration.maxInputTokens else {
                throw LogitlyError.inputTooLong(actual: tokens.count, maximum: configuration.maxInputTokens)
            }
            rendered.append(answerBoundary)
            tokenRows.append(tokens)
            labelRows.append(try continuationTokenIDs(boundary: answerBoundary, baseTokens: tokens, labels: rowLabels))
        }

        let tokenCount = tokenRows.reduce(0) { $0 + $1.count }
        var batch = llama_batch_init(Int32(tokenCount), 0, 1)
        defer { llama_batch_free(batch) }
        batch.n_tokens = Int32(tokenCount)
        var inputIndex = 0
        var outputIndices: [Int32] = []
        for (sequence, tokens) in tokenRows.enumerated() {
            for (position, token) in tokens.enumerated() {
                batch.token[inputIndex] = token
                batch.pos[inputIndex] = Int32(position)
                batch.n_seq_id[inputIndex] = 1
                batch.seq_id[inputIndex]![0] = Int32(sequence)
                let final = position == tokens.count - 1
                batch.logits[inputIndex] = final ? 1 : 0
                if final { outputIndices.append(Int32(inputIndex)) }
                inputIndex += 1
            }
        }

        let memory = llama_get_memory(context)
        llama_memory_clear(memory, true)
        defer { llama_memory_clear(memory, true) }
        let started = DispatchTime.now().uptimeNanoseconds
        let status = llama_decode(context, batch)
        guard status == 0 else { throw LogitlyError.decodeFailed(status) }
        llama_synchronize(context)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000

        var restricted: [[Float]] = []
        for ((outputIndex, rowLabels), tokenIDs) in zip(zip(outputIndices, labels), labelRows) {
            guard let fullLogits = llama_get_logits_ith(context, outputIndex) else {
                throw LogitlyError.logitsUnavailable
            }
            restricted.append(try rowLabels.map { label in
                guard let token = tokenIDs[label] else { throw LogitlyError.invalidVerbalizer(label) }
                return fullLogits[Int(token)]
            })
        }
        return RuntimeBatch(
            logits: restricted,
            rendered: rendered,
            promptTokens: tokenRows,
            labelTokenIDs: labelRows,
            modelOnlySeconds: elapsed
        )
    }

    func unload() {
        if let context {
            llama_free(context)
            self.context = nil
        }
        if let model {
            llama_model_free(model)
            self.model = nil
        }
        vocab = nil
        template = nil
    }

    private func validateAllVerbalizers() throws {
        let probe = "Return exactly one label.\n\nAnswer:"
        let boundary = try render(probe)
        let tokens = try tokenize(boundary)
        let labels = Array(Set(profile.choiceLabels + profile.noulLabels + profile.scoreLabels)).sorted()
        _ = try continuationTokenIDs(boundary: boundary, baseTokens: tokens, labels: labels)
    }

    private func render(_ prompt: String) throws -> String {
        guard let template else { throw LogitlyError.chatTemplateMissing }
        return try "user".withCString { role in
            try prompt.withCString { content in
                var message = llama_chat_message(role: role, content: content)
                var buffer = [CChar](repeating: 0, count: max(8_192, prompt.utf8.count * 2 + 4_096))
                var length = buffer.withUnsafeMutableBufferPointer { pointer in
                    llama_chat_apply_template(template, &message, 1, true, pointer.baseAddress, Int32(pointer.count))
                }
                guard length >= 0 else { throw LogitlyError.unsupportedChatTemplate }
                if length >= buffer.count {
                    buffer = [CChar](repeating: 0, count: Int(length) + 1)
                    length = buffer.withUnsafeMutableBufferPointer { pointer in
                        llama_chat_apply_template(template, &message, 1, true, pointer.baseAddress, Int32(pointer.count))
                    }
                }
                guard length >= 0 else { throw LogitlyError.unsupportedChatTemplate }
                let rendered = String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
                if let expected = profile.assistantSuffix, !rendered.hasSuffix(expected) {
                    throw LogitlyError.unexpectedAnswerBoundary(expected: expected)
                }
                if let opening = rendered.range(of: "<think>", options: .backwards) {
                    let suffix = rendered[opening.upperBound...]
                    guard let closing = suffix.range(of: "</think>"),
                          suffix[..<closing.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        throw LogitlyError.unsupportedChatTemplate
                    }
                }
                return rendered
            }
        }
    }

    private func tokenize(_ text: String) throws -> [Int32] {
        guard let vocab else { throw LogitlyError.modelUnloaded }
        let data = Array(text.utf8)
        var tokens = [llama_token](repeating: 0, count: data.count + 8)
        var count = data.withUnsafeBytes { bytes in
            llama_tokenize(vocab, bytes.bindMemory(to: CChar.self).baseAddress, Int32(data.count), &tokens, Int32(tokens.count), false, true)
        }
        if count < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-count))
            count = data.withUnsafeBytes { bytes in
                llama_tokenize(vocab, bytes.bindMemory(to: CChar.self).baseAddress, Int32(data.count), &tokens, Int32(tokens.count), false, true)
            }
        }
        guard count >= 0 else { throw LogitlyError.invalidRequest("GGUF tokenization failed") }
        return Array(tokens.prefix(Int(count)))
    }

    private func continuationTokenIDs(boundary: String, baseTokens: [Int32], labels: [String]) throws -> [String: Int32] {
        var result: [String: Int32] = [:]
        for label in labels {
            let continued = try tokenize(boundary + label)
            guard continued.count == baseTokens.count + 1,
                  continued.dropLast().elementsEqual(baseTokens),
                  let token = continued.last else {
                throw LogitlyError.invalidVerbalizer(label)
            }
            result[label] = token
        }
        return result
    }

    private static func hasMetalDevice() -> Bool {
        for index in 0..<ggml_backend_dev_count() {
            guard let device = ggml_backend_dev_get(index) else { continue }
            let name = String(cString: ggml_backend_dev_name(device)).lowercased()
            let description = String(cString: ggml_backend_dev_description(device)).lowercased()
            if name.contains("metal") || description.contains("metal") { return true }
        }
        return false
    }
}
