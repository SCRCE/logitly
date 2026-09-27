import Foundation
import Logitly

struct Cell: Hashable, Codable, Sendable {
    let x: Int
    let y: Int
}

enum Direction: String, CaseIterable, Sendable, Codable {
    case up = "UP", right = "RIGHT", down = "DOWN", left = "LEFT"

    var delta: Cell {
        switch self {
        case .up: Cell(x: 0, y: -1)
        case .right: Cell(x: 1, y: 0)
        case .down: Cell(x: 0, y: 1)
        case .left: Cell(x: -1, y: 0)
        }
    }
}

struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed == 0 ? 0x9e3779b97f4a7c15 : seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }
}

struct SnakeGame: Sendable {
    let width = 8
    let height = 6
    let cycle: [Cell]
    let positions: [Cell: Int]
    private var random: SeededGenerator
    var body: [Cell]
    var food: Cell?
    var steps = 0
    var foodEaten = 0
    var history: [Cell]

    init(seed: UInt64 = 20_260_923) {
        var cells = (0..<8).map { Cell(x: $0, y: 0) }
        for y in 1..<6 {
            let xs = y.isMultiple(of: 2) ? Array(1..<8) : Array((1..<8).reversed())
            cells.append(contentsOf: xs.map { Cell(x: $0, y: y) })
        }
        cells.append(contentsOf: (1..<6).reversed().map { Cell(x: 0, y: $0) })
        cycle = cells
        positions = Dictionary(uniqueKeysWithValues: cells.enumerated().map { ($0.element, $0.offset) })
        random = SeededGenerator(seed: seed)
        body = [cells[3], cells[2], cells[1]]
        history = [cells[3]]
        food = nil
        food = nextFood()
    }

    func candidate(_ direction: Direction) -> Cell {
        Cell(x: body[0].x + direction.delta.x, y: body[0].y + direction.delta.y)
    }

    func safeMoves() -> [Direction: Bool] {
        let head = positions[body[0]]!
        let tailGap = (positions[body.last!]! - head + cycle.count) % cycle.count
        return Dictionary(uniqueKeysWithValues: Direction.allCases.map { direction in
            let cell = candidate(direction)
            guard let candidatePosition = positions[cell] else { return (direction, false) }
            let occupied = body.dropLast().contains(cell) || (cell == body.last && cell == food)
            let advance = (candidatePosition - head + cycle.count) % cycle.count
            return (direction, !occupied && advance > 0 && advance < tailGap)
        })
    }

    func plannerDirection() -> Direction {
        let safe = safeMoves()
        let recent = Set(history.suffix(8))
        return Direction.allCases.filter { safe[$0] == true }.min { lhs, rhs in
            let leftCell = candidate(lhs)
            let rightCell = candidate(rhs)
            let leftCost = (positions[food!]! - positions[leftCell]! + cycle.count) % cycle.count
            let rightCost = (positions[food!]! - positions[rightCell]! + cycle.count) % cycle.count
            if leftCost != rightCost { return leftCost < rightCost }
            if recent.contains(leftCell) != recent.contains(rightCell) { return !recent.contains(leftCell) }
            return Direction.allCases.firstIndex(of: lhs)! < Direction.allCases.firstIndex(of: rhs)!
        }!
    }

    func decisionRequest() -> ChoiceRequest {
        let best = plannerDirection()
        let state = boardText() + "\nRecent head positions: " + history.suffix(8).map { "(\($0.x),\($0.y))" }.joined(separator: ", ")
        return ChoiceRequest(
            id: "snake-\(steps + 1)",
            state: .text(state),
            question: "The navigation planner recommends DIRECTION=\(best.rawValue). Select the choice with that exact DIRECTION value.",
            choices: Direction.allCases.map { ChoiceOption(id: $0.rawValue, text: "DIRECTION=\($0.rawValue)") }
        )
    }

    mutating func move(raw: Direction, probabilities: [String: Double]) -> (executed: Direction, shielded: Bool, ate: Bool) {
        let safe = safeMoves()
        let executed = safe[raw] == true ? raw : Direction.allCases.filter { safe[$0] == true }.max {
            (probabilities[$0.rawValue] ?? 0) < (probabilities[$1.rawValue] ?? 0)
        }!
        let cell = candidate(executed)
        let ate = cell == food
        body.insert(cell, at: 0)
        if ate {
            foodEaten += 1
            food = nextFood()
        } else {
            body.removeLast()
        }
        steps += 1
        history.append(cell)
        return (executed, raw != executed, ate)
    }

    func boardText() -> String {
        let bodyCells = Set(body)
        let rows = (0..<height).map { y in
            (0..<width).map { x -> Character in
                let cell = Cell(x: x, y: y)
                if cell == body[0] { return "H" }
                if bodyCells.contains(cell) { return "o" }
                if cell == food { return "*" }
                return "."
            }.map(String.init).joined()
        }.joined(separator: "\n")
        return "BOARD: 8 columns x 6 rows. (0,0) is top-left; x right, y down.\n    01234567\n" +
            rows.split(separator: "\n").enumerated().map { String(format: "%2d  %@", $0.offset, String($0.element)) }.joined(separator: "\n") +
            "\nLegend: H=head, o=body, *=food, .=empty."
    }

    private mutating func nextFood() -> Cell? {
        let free = cycle.filter { !body.contains($0) }
        guard !free.isEmpty else { return nil }
        return free[Int(random.next() % UInt64(free.count))]
    }
}

struct SnakeRecord: Identifiable, Sendable, Codable {
    let id: Int
    let raw: Direction
    let executed: Direction
    let shielded: Bool
    let ate: Bool
    let foodEaten: Int
    let probabilities: [String: Double]
    let seconds: Double
}
