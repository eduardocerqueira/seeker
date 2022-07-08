//date: 2022-07-08T16:54:17Z
//url: https://api.github.com/gists/27381d9e2d3f3ddfe7cd818db9247e5b
//owner: https://api.github.com/users/TeresaPlaz

package graphql_test

import (
  "context"
  "errors"
  
  . "github.com/onsi/ginkgo/v2"
  . "github.com/onsi/gomega"
  "github.com/golang/mock/gomock"
  "github.com/brianvoe/gofakeit/v6"
  
  "github.com/account/repo/middleware/internal/core/domain"
  "github.com/account/repo/middleware/internal/handlers/graphql"
)

var _ = Describe("Customer Queries", func() {
  defer GinkgoRecover()
  
  BeforeEach(func() {
    mockCtrl = gomock.NewController(GinkgoT())
    mockCustomerController = mocks.NewMockCustomerController(mockCtrl)
    resolver = graphql.NewResolver(mockCustomerController)
  })
  
  Context("Customer", func() {
    var mockId string
    var mockServiceOutput *domain.Customer
    
    BeforeEach(func() {
      mockId = gofakeit.UUID()
      gofakeit.Struct(&mockServiceOutput)
    })
    
    When("a customer id is received", func() {
      It("should return a customer from the controller's response", func() {
        mockCustomerController.EXPECT().Get(context.TODO(), mockId).Return(mockServiceOutput, nil).Times(1)
        result, err := resolver.Query().Customer(context.TODO(), mockId)
        Expect(result).To(Equal(mockServiceOutput))
        Expect(err).ToNot(HaveOccurred())
      })
    })
    
    When("The controller returns an error", func() {
      It("should return that error", func() {
        mockCustomerController.EXPECT().Get(context.TODO(), mockId).Return(nil, errors.New("Failed")).Times(1)
        result, err := resolver.Query().Customer(context.TODO(), mockId)
        Expect(result).To(BeNil())
        Expect(err).To(HaveOccurred())
        Expect(err.Error()).To(Equal("Failed"))
      })
    })
  })
})