import Foundation

enum ContractLimits {
    static let stateCharacters = 131_072
    static let questionCharacters = 16_384
    static let criterionCharacters = 16_384
    static let identifierCharacters = 256
}

enum StateSerializer {
    static func serialize(_ state: DecisionState) throws -> String {
        let rendered: String
        switch state {
        case let .text(text): rendered = text
        case let .json(value): rendered = try renderJSON(value)
        }
        guard !rendered.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LogitlyError.invalidRequest("state must not be empty")
        }
        guard rendered.count <= ContractLimits.stateCharacters else {
            throw LogitlyError.invalidRequest("state exceeds \(ContractLimits.stateCharacters) characters")
        }
        return rendered
    }

    static func text(_ value: String, name: String, maximum: Int) throws -> String {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { throw LogitlyError.invalidRequest("\(name) must be a non-empty string") }
        guard value.count <= maximum else { throw LogitlyError.invalidRequest("\(name) exceeds \(maximum) characters") }
        return normalized
    }

    private static func renderJSON(_ value: JSONValue) throws -> String {
        switch value {
        case .null: return "null"
        case let .bool(value): return value ? "true" : "false"
        case let .integer(value): return String(value)
        case let .number(value):
            guard value.isFinite else { throw LogitlyError.invalidRequest("state contains a non-finite number") }
            let data = try encoder().encode(value)
            return String(decoding: data, as: UTF8.self)
        case let .string(value):
            let data = try encoder().encode(value)
            return String(decoding: data, as: UTF8.self)
        case let .array(values):
            return "[" + (try values.map(renderJSON).joined(separator: ",")) + "]"
        case let .object(values):
            return "{" + (try values.keys.sorted().map { key in
                let encodedKey = try renderJSON(.string(key))
                return encodedKey + ":" + (try renderJSON(values[key]!))
            }.joined(separator: ",")) + "}"
        }
    }

    private static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return encoder
    }
}
