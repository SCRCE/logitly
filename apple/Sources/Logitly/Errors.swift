import Foundation

public enum LogitlyError: Error, LocalizedError, Sendable, Equatable {
    case modelNotFound(String)
    case unsupportedSimulator
    case metalUnavailable
    case modelLoadFailed
    case contextCreationFailed
    case chatTemplateMissing
    case unsupportedChatTemplate
    case unexpectedAnswerBoundary(expected: String)
    case invalidVerbalizer(String)
    case invalidRequest(String)
    case inputTooLong(actual: Int, maximum: Int)
    case decodeFailed(Int32)
    case logitsUnavailable
    case modelUnloaded

    public var errorDescription: String? {
        switch self {
        case let .modelNotFound(path): "Model not found at \(path)."
        case .unsupportedSimulator: "Real model inference requires a physical Apple device."
        case .metalUnavailable: "The llama.cpp Metal backend is unavailable."
        case .modelLoadFailed: "llama.cpp could not load the GGUF model."
        case .contextCreationFailed: "llama.cpp could not create the inference context."
        case .chatTemplateMissing: "The GGUF does not contain a chat template."
        case .unsupportedChatTemplate: "llama.cpp cannot render this GGUF chat template."
        case let .unexpectedAnswerBoundary(expected): "The prompt does not end at the expected answer boundary: \(expected.debugDescription)."
        case let .invalidVerbalizer(label): "Label \(label.debugDescription) is not exactly one token at the answer boundary."
        case let .invalidRequest(message): message
        case let .inputTooLong(actual, maximum): "Rendered input has \(actual) tokens; the maximum is \(maximum)."
        case let .decodeFailed(code): "llama_decode failed with status \(code)."
        case .logitsUnavailable: "llama.cpp returned no final-position logits."
        case .modelUnloaded: "The decision model has been unloaded."
        }
    }
}
