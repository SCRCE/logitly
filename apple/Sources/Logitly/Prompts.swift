import Foundation

enum PromptRenderer {
    static func choice(state: String, question: String, labels: [String], choices: [String]) -> String {
        let options = zip(labels, choices).map { "\($0). \($1)" }.joined(separator: "\n")
        let allowed = labels.joined(separator: "\n")
        return "STATE:\n\(state)\n\nQUESTION:\n\(question)\n\nCHOICES:\n\(options)\n\nReturn exactly one label from:\n\(allowed)\n\nAnswer:"
    }

    static func noul(state: String, question: String) -> String {
        "STATE:\n\(state)\n\nQUESTION:\n\(question)\n\nChoose exactly one:\n\nY = Yes\nN = No\n\nReturn exactly one label from:\nY\nN\n\nAnswer:"
    }

    static func score(state: String, question: String, labels: [String], levels: [String]) -> String {
        let options = zip(labels, levels).map { "\($0). \($1)" }.joined(separator: "\n")
        let allowed = labels.joined(separator: "\n")
        return "STATE:\n\(state)\n\nQUESTION:\n\(question)\n\nORDERED LEVELS:\n\(options)\n\nReturn exactly one level label from:\n\(allowed)\n\nAnswer:"
    }
}

enum RestrictedSoftmax {
    static func probabilities(_ logits: [Float]) throws -> [Double] {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite) else {
            throw LogitlyError.logitsUnavailable
        }
        let maximum = logits.max()!
        let exponentials = logits.map { exp(Double($0 - maximum)) }
        let total = exponentials.reduce(0, +)
        guard total.isFinite, total > 0 else { throw LogitlyError.logitsUnavailable }
        return exponentials.map { $0 / total }
    }
}

struct PreparedDecision: Sendable {
    enum Kind: Sendable { case noul, choice, score }
    let id: String
    let kind: Kind
    let prompt: String
    let labels: [String]
    let semanticIDs: [String]
}
