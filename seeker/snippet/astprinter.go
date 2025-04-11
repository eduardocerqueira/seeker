//date: 2025-04-11T16:47:39Z
//url: https://api.github.com/gists/8ce1413736ea38cf3e9d7cb6c7268b27
//owner: https://api.github.com/users/admin-else

package main

import (
	"fmt"
	"glox/glox"
)

type printer struct{}

func (p printer) VisitBinrayExpr(e glox.BinaryExpr) any {
	return nil
}

func (p printer) VisitGroupingExpr(e glox.GroupingExpr) any {
	return fmt.Sprintf("(%v)", e.Expr.Accept(p))
}

func (p printer) VisitLiteralExpr(e glox.LiteralExpr) any {
	return fmt.Sprintf("%v", e.Value)
}

func (p printer) VisitUnaryExpr(e glox.UnaryExpr) any {
	return nil
}

func (p printer) print(e glox.Expr) any {
	return e.Accept(p)
}

func main() {
	p := printer{}
	fmt.Println(p.print(glox.GroupingExpr{Expr: glox.LiteralExpr{Value: 123}}))
}

