//date: 2022-03-03T16:57:01Z
//url: https://api.github.com/gists/99358e2529a5887ae13a936a7f1c8ae5
//owner: https://api.github.com/users/qdm12

package potato

import (
	"errors"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

//go:generate mockgen -destination=mock_cutter_test.go -package $GOPACKAGE . Cutter
//go:generate mockgen -destination=mock_fryer_test.go -package $GOPACKAGE . Fryer

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
			ctrl := gomock.NewController(t)

			cutter := NewMockCutter(ctrl)
			var previousCall *gomock.Call
			for _, call := range testCase.cutCalls {
				newCall := cutter.EXPECT().Cut(call.potato).
					Return(call.uncookedFries, call.err)
				if previousCall != nil {
					newCall.After(previousCall)
				}
				previousCall = newCall
			}

			fryer := NewMockFryer(ctrl)
			for _, call := range testCase.fryCalls {
				fryer.EXPECT().Fry(call.uncookedFries).
					Return(call.cookedFries, call.err)
			}

			fries, err := CutAndFry(cutter, fryer, testCase.potatoes)

			assert.Equal(t, testCase.expectedFries, fries)
			assert.ErrorIs(t, err, testCase.expectedErr)
			if err != nil {
				assert.EqualError(t, err, testCase.expectedErrMessage)
			}
		})
	}
}

//go:generate mockgen -destination=mock_fetcher_test.go -package $GOPACKAGE . Fetcher

// ==================================
// ==================================
// ==================================
// Part 2 mocks returning other mocks
// ==================================
// ==================================
// ==================================

func Test_MakeFries_Table(t *testing.T) {
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
		fetchErr           error // NEW
		potatoes           []Potato
		expectedFries      []Frie
		expectedErr        error
		expectedErrMessage string
	}{
		"fetch error": {
			fetchErr:           errTest,
			expectedErr:        errTest,
			expectedErrMessage: "cannot get our tools: test sentinel error",
		},
		"success for one potato": {
			cutCalls: []cutCall{
				{
					potato:        APotato,
					uncookedFries: []Frie{{Cooked: false}, {Cooked: false}},
				},
			},
			fryCalls: []fryCall{
				{
					uncookedFries: []Frie{{Cooked: false}, {Cooked: false}},
					cookedFries:   []Frie{{Cooked: true}, {Cooked: true}},
				},
			},
			potatoes:      []Potato{APotato},
			expectedFries: []Frie{{Cooked: true}, {Cooked: true}}},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			ctrl := gomock.NewController(t)

			cutter := NewMockCutter(ctrl)
			for _, call := range testCase.cutCalls {
				cutter.EXPECT().Cut(call.potato).Return(call.uncookedFries, call.err)
			}

			fryer := NewMockFryer(ctrl)
			for _, call := range testCase.fryCalls {
				fryer.EXPECT().Fry(call.uncookedFries).Return(call.cookedFries, call.err)
			}

			fetcher := NewMockFetcher(ctrl)
			fetcher.EXPECT().FetchTools().
				Return(cutter, fryer, testCase.fetchErr)

			fries, err := MakeFries(fetcher, testCase.potatoes)

			assert.Equal(t, testCase.expectedFries, fries)
			assert.ErrorIs(t, err, testCase.expectedErr)
			if err != nil {
				assert.EqualError(t, err, testCase.expectedErrMessage)
			}
		})
	}
}

func Test_MakeFries_SimpleSubtests(t *testing.T) {
	errTest := errors.New("test sentinel error")

	t.Run("fetch error", func(t *testing.T) {
		ctrl := gomock.NewController(t)

		fetcher := NewMockFetcher(ctrl)
		fetcher.EXPECT().FetchTools().
			Return(nil, nil, errTest)

		fries, err := MakeFries(fetcher, nil)

		assert.Nil(t, fries)
		assert.ErrorIs(t, err, errTest)
		if err != nil {
			assert.EqualError(t, err, "cannot get our tools: test sentinel error")
		}
	})

	t.Run("success for one potato", func(t *testing.T) {
		ctrl := gomock.NewController(t)

		cutter := NewMockCutter(ctrl)
		cutter.EXPECT().Cut(APotato).
			Return([]Frie{{Cooked: false}, {Cooked: false}}, nil)

		fryer := NewMockFryer(ctrl)
		fryer.EXPECT().Fry([]Frie{{Cooked: false}, {Cooked: false}}).
			Return([]Frie{{Cooked: true}, {Cooked: true}}, nil)

		fetcher := NewMockFetcher(ctrl)
		fetcher.EXPECT().FetchTools().
			Return(cutter, fryer, nil)

		fries, err := MakeFries(fetcher, []Potato{APotato})

		expectedFries := []Frie{{Cooked: true}, {Cooked: true}}
		assert.Equal(t, expectedFries, fries)
		assert.NoError(t, err)
	})
}

func Test_MakeFries_Table_Functions(t *testing.T) {
	errTest := errors.New("test sentinel error")

	testCases := map[string]struct {
		cutterBuilder      func(ctrl *gomock.Controller) Cutter
		fryerBuilder       func(ctrl *gomock.Controller) Fryer
		fetcherBuilder     func(ctrl *gomock.Controller, cutter Cutter, fryer Fryer) Fetcher
		potatoes           []Potato
		expectedFries      []Frie
		expectedErr        error
		expectedErrMessage string
	}{
		"fetch error": {
			cutterBuilder: func(ctrl *gomock.Controller) Cutter { return nil },
			fryerBuilder:  func(ctrl *gomock.Controller) Fryer { return nil },
			fetcherBuilder: func(ctrl *gomock.Controller, cutter Cutter, fryer Fryer) Fetcher {
				fetcher := NewMockFetcher(ctrl)
				fetcher.EXPECT().FetchTools().Return(nil, nil, errTest)
				return fetcher
			},
			expectedErr:        errTest,
			expectedErrMessage: "cannot get our tools: test sentinel error",
		},
		"success for 1 potato": {
			cutterBuilder: func(ctrl *gomock.Controller) Cutter {
				cutter := NewMockCutter(ctrl)
				cutter.EXPECT().Cut(APotato).
					Return([]Frie{{Cooked: false}, {Cooked: false}}, nil)
				return cutter
			},
			fryerBuilder: func(ctrl *gomock.Controller) Fryer {
				fryer := NewMockFryer(ctrl)
				fryer.EXPECT().Fry([]Frie{{Cooked: false}, {Cooked: false}}).
					Return([]Frie{{Cooked: true}, {Cooked: true}}, nil)
				return fryer
			},
			fetcherBuilder: func(ctrl *gomock.Controller, cutter Cutter, fryer Fryer) Fetcher {
				fetcher := NewMockFetcher(ctrl)
				fetcher.EXPECT().FetchTools().Return(cutter, fryer, nil)
				return fetcher
			},
			potatoes:      []Potato{APotato},
			expectedFries: []Frie{{Cooked: true}, {Cooked: true}}},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			ctrl := gomock.NewController(t)

			cutter := testCase.cutterBuilder(ctrl)
			fryer := testCase.fryerBuilder(ctrl)
			fetcher := testCase.fetcherBuilder(ctrl, cutter, fryer)

			fries, err := MakeFries(fetcher, testCase.potatoes)

			assert.Equal(t, testCase.expectedFries, fries)
			assert.ErrorIs(t, err, testCase.expectedErr)
			if err != nil {
				assert.EqualError(t, err, testCase.expectedErrMessage)
			}
		})
	}
}

func bigFunc(n int, f func(n int) int) int {
	x := f(n)
	return x * 2
}

func Test_bigFunc(t *testing.T) {
	timesCalled := 0
	f := func(n int) int {
		timesCalled++
		return 1
	}
	x := bigFunc(2, f)
	assert.Equal(t, 4, x)
}
