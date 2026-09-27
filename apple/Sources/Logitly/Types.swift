import Foundation

public enum JSONValue: Sendable, Equatable, Codable {
    case null
    case bool(Bool)
    case integer(Int64)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let box = try decoder.singleValueContainer()
        if box.decodeNil() { self = .null }
        else if let value = try? box.decode(Bool.self) { self = .bool(value) }
        else if let value = try? box.decode(Int64.self) { self = .integer(value) }
        else if let value = try? box.decode(Double.self) { self = .number(value) }
        else if let value = try? box.decode(String.self) { self = .string(value) }
        else if let value = try? box.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try box.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var box = encoder.singleValueContainer()
        switch self {
        case .null: try box.encodeNil()
        case let .bool(value): try box.encode(value)
        case let .integer(value): try box.encode(value)
        case let .number(value): try box.encode(value)
        case let .string(value): try box.encode(value)
        case let .array(value): try box.encode(value)
        case let .object(value): try box.encode(value)
        }
    }
}

extension JSONValue: ExpressibleByNilLiteral {
    public init(nilLiteral: ()) { self = .null }
}

extension JSONValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .bool(value) }
}

extension JSONValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int64) { self = .integer(value) }
}

extension JSONValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) { self = .number(value) }
}

extension JSONValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
}

extension JSONValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
}

extension JSONValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        var object: [String: JSONValue] = [:]
        for (key, value) in elements { object[key] = value }
        self = .object(object)
    }
}

public enum DecisionState: Sendable, Equatable {
    case text(String)
    case json(JSONValue)
}

public struct ChoiceOption: Sendable, Equatable, Identifiable {
    public let id: String
    public let text: String

    public init(id: String, text: String) {
        self.id = id
        self.text = text
    }
}

public struct NoulRequest: Sendable, Equatable {
    public let id: String
    public let state: DecisionState
    public let question: String

    public init(id: String = "noul", state: DecisionState, question: String) {
        self.id = id
        self.state = state
        self.question = question
    }
}

public struct ChoiceRequest: Sendable, Equatable {
    public let id: String
    public let state: DecisionState
    public let question: String
    public let choices: [ChoiceOption]

    public init(id: String = "choice", state: DecisionState, question: String, choices: [ChoiceOption]) {
        self.id = id
        self.state = state
        self.question = question
        self.choices = choices
    }
}

public struct ScoreRequest: Sendable, Equatable {
    public let id: String
    public let state: DecisionState
    public let question: String
    public let levels: [String]

    public init(id: String = "score", state: DecisionState, question: String, levels: [String]) {
        self.id = id
        self.state = state
        self.question = question
        self.levels = levels
    }
}

public enum DecisionRequest: Sendable, Equatable {
    case noul(NoulRequest)
    case choice(ChoiceRequest)
    case score(ScoreRequest)
}

public struct NoulResult: Sendable, Equatable, Codable {
    public let yes: Double
    public let no: Double
}

public struct ChoiceResult: Sendable, Equatable, Codable {
    public let probabilities: [String: Double]
    public let choice: String
    public let confidence: Double
}

public struct ScoreResult: Sendable, Equatable, Codable {
    public let distribution: [Double]
    public let score: Double
    public let confidence: Double
}

public enum DecisionResult: Sendable, Equatable {
    case noul(id: String, NoulResult)
    case choice(id: String, ChoiceResult)
    case score(id: String, ScoreResult)
}

public struct PromptInspection: Sendable, Equatable {
    public let prompt: String
    public let renderedPrompt: String
    public let promptTokenIDs: [Int32]
    public let labelTokenIDs: [String: Int32]
}

public struct InferenceDiagnostics: Sendable, Equatable {
    public let backend: String
    public let modelOnlySeconds: Double
    public let generatedTokens: Int
}

public struct ModelConfiguration: Sendable, Equatable {
    public var maxInputTokens: Int
    public var microBatchTokens: Int
    public var maxBatchSize: Int
    public var threads: Int
    public var requireMetal: Bool

    public init(
        maxInputTokens: Int = 1_024,
        microBatchTokens: Int = 256,
        maxBatchSize: Int = 8,
        threads: Int = 0,
        requireMetal: Bool = true
    ) {
        self.maxInputTokens = maxInputTokens
        self.microBatchTokens = microBatchTokens
        self.maxBatchSize = maxBatchSize
        self.threads = threads
        self.requireMetal = requireMetal
    }

    public static let iphoneDefault = ModelConfiguration()
}
