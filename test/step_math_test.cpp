// Host test for StepMath.h — the forward step-distance helper that
// SplitFlapDisplay::moveTo() uses to track how many steps each module still
// owes. Includes the REAL production header; no re-implementation.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -I src test/step_math_test.cpp -o /tmp/step_math_test && /tmp/step_math_test

#include "StepMath.h"

#include <cstdio>

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (! (cond)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

static void test_forward_steps() {
    const int sp = 2048;

    // Zero distance in both trivial forms.
    CHECK(forwardSteps(0, 0, sp) == 0);
    CHECK(forwardSteps(5, 5, sp) == 0);
    CHECK(forwardSteps(sp - 1, sp - 1, sp) == 0);

    // Plain forward moves.
    CHECK(forwardSteps(0, 1, sp) == 1);
    CHECK(forwardSteps(0, 5, sp) == 5);
    CHECK(forwardSteps(100, 200, sp) == 100);

    // Wrap-around: the drum only turns forwards, so 2047 -> 0 is one step.
    CHECK(forwardSteps(sp - 1, 0, sp) == 1);
    CHECK(forwardSteps(sp - 3, 2, sp) == 5);
    CHECK(forwardSteps(0, sp - 1, sp) == sp - 1);

    // Backwards-looking target = almost a full revolution.
    CHECK(forwardSteps(1, 0, sp) == sp - 1);
    CHECK(forwardSteps(100, 99, sp) == sp - 1);

    // Inputs already wrapped (defensive: caller should pass normalized
    // positions, but the helper must not go negative on from > sp).
    CHECK(forwardSteps(sp + 5, 10, sp) == 5);
    CHECK(forwardSteps(-1, 0, sp) == 1);
}

static void test_forward_steps_other_sizes() {
    // 37-char charset uses a different stepsPerRot in theory; the helper is
    // size-agnostic. Small sizes exercise the modulo paths hard.
    CHECK(forwardSteps(0, 0, 37) == 0);
    CHECK(forwardSteps(36, 0, 37) == 1);
    CHECK(forwardSteps(0, 36, 37) == 36);
    CHECK(forwardSteps(2, 1, 37) == 36);
    CHECK(forwardSteps(0, 0, 1) == 0);
    CHECK(forwardSteps(0, 0, 48) == 0);
    CHECK(forwardSteps(47, 0, 48) == 1);
}

int main() {
    test_forward_steps();
    test_forward_steps_other_sizes();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
