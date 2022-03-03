//date: 2022-03-03T16:57:01Z
//url: https://api.github.com/gists/99358e2529a5887ae13a936a7f1c8ae5
//owner: https://api.github.com/users/qdm12

package potato

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

//go:generate mockery --name Cutter --case underscore --inpackage --testonly
//go:generate mockery --name Fryer --case underscore --inpackage --testonly

func Test_CutAndFry(t *testing.T) {
	errTest := errors.New("test sentinel error")

	type cutCall struct {
		potato        Potato
		uncookedFries []Frie
		err           error
	}

	type fryCall struct {
		uncookedFries []Frie
		cookedFries   []Frie
		err           error
	}

	testCases := map[string]struct {
		cutCalls           []cutCall
		fryCalls           []fryCall
		potatoes           []Potato
		expectedFries      []Frie
		expectedErr        error
		expectedErrMessage string
	}{
		"cut error": {
			cutCalls: []cutCall{
				{potato: APotato, err: errTest},
			},
			potatoes:           []Potato{APotato},
			expectedErr:        errTest,
			expectedErrMessage: "cannot cut potato: test sentinel error",
		},
		"success for 2 potatoes": {
			cutCalls: []cutCall{
				{
					potato:        APotato,
					uncookedFries: []Frie{{Cooked: false}, {Cooked: false}},
				},
				{
					potato:        APotato,
					uncookedFries: []Frie{{Cooked: false}, {Cooked: false}, {Cooked: false}},
				},
			},
			fryCalls: []fryCall{
				{
					uncookedFries: []Frie{{Cooked: false}, {Cooked: false}},
					cookedFries:   []Frie{{Cooked: true}, {Cooked: true}},
				},
				{
					uncookedFries: []Frie{{Cooked: false}, {Cooked: false}, {Cooked: false}},
					cookedFries:   []Frie{{Cooked: true}, {Cooked: true}, {Cooked: true}},
				},
			},
			potatoes:      []Potato{APotato, APotato},
			expectedFries: []Frie{{Cooked: true}, {Cooked: true}, {Cooked: true}, {Cooked: true}, {Cooked: true}},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			// Using Mockery: START ======================
			cutter := new(MockCutter)
			for _, call := range testCase.cutCalls {
				cutter.On("Cut", call.potato).Return(call.uncookedFries, call.err).Once()
				// - Method call is a string so it's untyped - hard for type changes in codebase
				// - You need to call .Once() for it to expect it once, imo bad default
			}

			fryer := new(MockFryer)
			for _, call := range testCase.fryCalls {
				fryer.On("Fry", call.uncookedFries).Return(call.cookedFries, call.err).Once()
			}
			// Using Mockery: END ======================

			fries, err := CutAndFry(cutter, fryer, testCase.potatoes)

			assert.Equal(t, testCase.expectedFries, fries)
			assert.ErrorIs(t, err, testCase.expectedErr)
			if err != nil {
				assert.EqualError(t, err, testCase.expectedErrMessage)
			}

			// Mockery additional step required (EASY TO FORGET)
			cutter.AssertExpectations(t)
			fryer.AssertExpectations(t)
		})
	}
}
