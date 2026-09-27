import XCTest
@testable import Logitly

final class ContractTests: XCTestCase {
    func testDeterministicJSONMatchesPythonContract() throws {
        let state: DecisionState = .json([
            "z": .array([.integer(2), .bool(true), .null]),
            "a": .object(["message": .string("hello")]),
        ])
        XCTAssertEqual(try StateSerializer.serialize(state), #"{"a":{"message":"hello"},"z":[2,true,null]}"#)
        XCTAssertEqual(
            try StateSerializer.serialize(.json(["url": "https://logitly.ai/مرحبا"])),
            #"{"url":"https://logitly.ai/مرحبا"}"#
        )
    }

    func testPromptGoldenStrings() {
        XCTAssertEqual(
            PromptRenderer.choice(state: "case", question: "Act?", labels: ["A", "B"], choices: ["Approve", "Review"]),
            "STATE:\ncase\n\nQUESTION:\nAct?\n\nCHOICES:\nA. Approve\nB. Review\n\nReturn exactly one label from:\nA\nB\n\nAnswer:"
        )
        XCTAssertEqual(
            PromptRenderer.noul(state: "case", question: "Valid?"),
            "STATE:\ncase\n\nQUESTION:\nValid?\n\nChoose exactly one:\n\nY = Yes\nN = No\n\nReturn exactly one label from:\nY\nN\n\nAnswer:"
        )
    }

    func testRestrictedSoftmaxNeverNormalizesAcrossOtherTokens() throws {
        let values = try RestrictedSoftmax.probabilities([10.2, 13.8, 11.1])
        XCTAssertEqual(values.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertGreaterThan(values[1], 0.9)
        XCTAssertEqual(try RestrictedSoftmax.probabilities([10_000, 10_000]), [0.5, 0.5])
    }

    func testPrimitivesAndExpectedScore() async throws {
        let runtime = FakeRuntime(rows: [
            [2, 1],
            [1, 4, 2],
            [0, 1, 3],
        ])
        let model = DecisionModel(testRuntime: runtime)
        let requests: [DecisionRequest] = [
            .noul(NoulRequest(state: .text("s"), question: "q")),
            .choice(ChoiceRequest(state: .text("s"), question: "q", choices: [
                ChoiceOption(id: "a", text: "A"), ChoiceOption(id: "b", text: "B"), ChoiceOption(id: "c", text: "C"),
            ])),
            .score(ScoreRequest(state: .text("s"), question: "q", levels: ["low", "medium", "high"])),
        ]
        let results = try await model.decideMany(requests, batchSize: 3)
        guard case let .noul(_, noul) = results[0],
              case let .choice(_, choice) = results[1],
              case let .score(_, score) = results[2] else {
            return XCTFail("Wrong result types")
        }
        XCTAssertGreaterThan(noul.yes, noul.no)
        XCTAssertEqual(choice.choice, "b")
        XCTAssertEqual(score.score, score.distribution[1] + 2 * score.distribution[2], accuracy: 1e-12)
        XCTAssertEqual(runtime.calls, 1)
    }

    func testValidationRejectsDuplicatesAndOversizedChoiceSets() async throws {
        let model = DecisionModel(testRuntime: FakeRuntime(rows: []))
        await XCTAssertThrowsErrorAsync {
            _ = try await model.choice(state: .text("s"), question: "q", choices: [
                ChoiceOption(id: "same", text: "one"), ChoiceOption(id: "same", text: "two"),
            ])
        }
        await XCTAssertThrowsErrorAsync {
            _ = try await model.choice(
                state: .text("s"),
                question: "q",
                choices: (0..<21).map { ChoiceOption(id: "\($0)", text: "\($0)") }
            )
        }
    }
}

private final class FakeRuntime: DirectLogitRuntime, @unchecked Sendable {
    let backendName = "fake"
    private let rows: [[Float]]
    private(set) var calls = 0

    init(rows: [[Float]]) { self.rows = rows }

    func score(prompts: [String], labels: [[String]]) throws -> RuntimeBatch {
        calls += 1
        let selected = Array(rows.prefix(prompts.count))
        return RuntimeBatch(
            logits: selected,
            rendered: prompts.map { $0 + "<assistant>" },
            promptTokens: prompts.map { _ in [1, 2, 3] },
            labelTokenIDs: labels.map { Dictionary(uniqueKeysWithValues: $0.enumerated().map { ($0.element, Int32($0.offset + 10)) }) },
            modelOnlySeconds: 0.001
        )
    }

    func unload() {}
}

private func XCTAssertThrowsErrorAsync(
    _ expression: () async throws -> Void,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        try await expression()
        XCTFail("Expected error", file: file, line: line)
    } catch {}
}
