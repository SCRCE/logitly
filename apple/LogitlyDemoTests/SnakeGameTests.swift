import XCTest
@testable import LogitlyDemo

final class SnakeGameTests: XCTestCase {
    func testPlannerAlwaysReturnsSafeMove() {
        var game = SnakeGame()
        for _ in 0..<50 {
            let direction = game.plannerDirection()
            XCTAssertEqual(game.safeMoves()[direction], true)
            _ = game.move(raw: direction, probabilities: [direction.rawValue: 1])
        }
    }
}
