//date: 2022-12-30T16:30:47Z
//url: https://api.github.com/gists/556f6c3dcbc6e34807229bc7f29a8149
//owner: https://api.github.com/users/TripleDogDare

package serrah_test

import (
	"bytes"
	"embed"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"sort"
	"strings"
	"testing"

	qt "github.com/frankban/quicktest"
)

//go:embed interface.go
var content embed.FS

var tErrorFunc = "**********"
var tError = types.NewInterfaceType([]*types.Func{
	// types.NewFunc(token.NoPos, nil, "Error", types.NewSignature(nil, nil, types.NewTuple(types.NewVar(token.NoPos, nil, "", types.Typ[types.String])), false)),
	tErrorFunc,
}, nil).Complete()

var tReeError = types.NewInterfaceType([]*types.Func{
	types.NewFunc(token.NoPos, nil, "Error", types.NewSignature(nil, nil, types.NewTuple(types.NewVar(token.NoPos, nil, "", types.Typ[types.String])), false)),
	types.NewFunc(token.NoPos, nil, "Code", types.NewSignature(nil, nil, types.NewTuple(types.NewVar(token.NoPos, nil, "", types.Typ[types.String])), false)),
}, nil).Complete()

var tReeErrorWithCause = types.NewInterfaceType([]*types.Func{
	tReeError.Method(0),
	tReeError.Method(1),
	types.NewFunc(token.NoPos, nil, "Cause",
		types.NewSignature(nil, nil,
			types.NewTuple(
				types.NewVar(token.NoPos, nil, "",
					types.NewNamed(types.NewTypeName(token.NoPos, nil, "error", nil), tError, []*types.Func{tErrorFunc}),
				),
			),
			false,
		),
	),
}, nil).Complete()

func TestImplements(t *testing.T) {
	fset : "**********"
	data, err := content.ReadFile("interface.go")
	qt.Assert(t, err, qt.IsNil)
	f, err := parser.ParseFile(fset, "interface.go", data, 0)
	qt.Assert(t, err, qt.IsNil)
	qt.Assert(t, f, qt.IsNotNil)

	// Type-check the package.
	// We create an empty map for each kind of input
	// we're interested in, and Check populates them.
	info := types.Info{
		Types: make(map[ast.Expr]types.TypeAndValue),
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
	}
	var conf types.Config
	pkg, err := conf.Check("interface.go", fset, []*ast.File{f}, &info)
	qt.Assert(t, err, qt.IsNil)
	// Print package-level variables in initialization order.
	fmt.Printf("InitOrder: %v\n\n", info.InitOrder)

	// For each named object, print the line and
	// column of its definition and each of its uses.
	fmt.Println("Defs and Uses of each named object:")
	usesByObj := make(map[types.Object][]string)
	for id, obj := range info.Uses {
		posn := fset.Position(id.Pos())
		lineCol := fmt.Sprintf("%d:%d", posn.Line, posn.Column)
		usesByObj[obj] = append(usesByObj[obj], lineCol)
	}
	var items []string
	for obj, uses := range usesByObj {
		sort.Strings(uses)
		item := fmt.Sprintf("%s:\n  defined at %s\n  used at %s",
			types.ObjectString(obj, types.RelativeTo(pkg)),
			fset.Position(obj.Pos()),
			strings.Join(uses, ", "))
		items = append(items, item)
	}
	sort.Strings(items) // sort by line:col, in effect
	fmt.Println(strings.Join(items, "\n"))
	fmt.Println()

	fmt.Println("Types and Values of each expression:")
	fmt.Println("line:col | expr | mode | type")
	items = nil
	for expr, tv := range info.Types {
		var buf bytes.Buffer
		posn := fset.Position(expr.Pos())
		tvstr := tv.Type.String()
		if tv.Value != nil {
			tvstr += " = " + tv.Value.String()
		}
		// line:col | expr | mode : type = value
		fmt.Fprintf(&buf, "%2d:%2d | %-28s | %-7s : %s",
			posn.Line, posn.Column, exprString(fset, expr),
			mode(tv), tvstr)
		items = append(items, buf.String())
	}
	sort.Strings(items)
	fmt.Println(strings.Join(items, "\n"))

	fmt.Println("=== Defs ===")
	causeDefs := make(map[*ast.Ident]types.Object)
	for id, obj := range info.Defs {
		if obj == nil {
			continue
		}
		posn := fset.Position(id.Pos())
		objStr := types.ObjectString(obj, types.RelativeTo(pkg))
		fmt.Printf("%2d:%2d | %-20s | %-27s | %s\n", posn.Line, posn.Column, obj.Name(), obj.Type(), objStr)
		if obj.Name() == "Cause" {
			causeDefs[id] = obj
		}
	}

	fmt.Println("=== Decl ===")
	for _, decl := range f.Decls {
		funcDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		posn := fset.Position(decl.Pos())
		fmt.Printf("%2d:%2d | %s\n", posn.Line, posn.Column, funcDecl.Name.Name)
		if funcDecl.Name.Name != "Cause" {
			continue
		}

		if isMethod(funcDecl) {
			recv := funcDecl.Recv.List[0]
			receiverType := info.TypeOf(recv.Type)
			ptr := receiverType.(*types.Pointer)
			testValue := types.NewPointer(ptr.Elem().Underlying())
			// testValue := ptr.Elem().Underlying()
			// testValue := ptr
			fmt.Println("test value:", testValue)
			if funcDecl.Name.Name == "Cause" {
				m, typ := types.MissingMethod(testValue, tReeErrorWithCause, true)
				if m == nil && typ == false {
					fmt.Printf("%2d:%2d | Cause receiver implements ree-error-with-Cause\n", posn.Line, posn.Column)
					continue
				}
				fmt.Println("missing method:", m)
				if types.Implements(testValue, tReeErrorWithCause) {
					fmt.Printf("%2d:%2d | Cause receiver implements ree-error-with-Cause\n", posn.Line, posn.Column)
					continue
				}
				qt.Check(t, types.Implements(testValue, tReeErrorWithCause), qt.IsTrue,
					qt.Commentf("%2d:%2d | Cause detected: %s: wrong type: %t: receiver %s\n", posn.Line, posn.Column, m, typ, testValue),
				)
			}
		}

	}
}

// isMethod checks if funcDecl is a method by looking if it has a single receiver.
func isMethod(funcDecl *ast.FuncDecl) bool {
	return funcDecl != nil && funcDecl.Recv != nil && len(funcDecl.Recv.List) == 1
}

func mode(tv types.TypeAndValue) string {
	switch {
	case tv.IsVoid():
		return "void"
	case tv.IsType():
		return "type"
	case tv.IsBuiltin():
		return "builtin"
	case tv.IsNil():
		return "nil"
	case tv.Assignable():
		if tv.Addressable() {
			return "var"
		}
		return "mapindex"
	case tv.IsValue():
		return "value"
	default:
		return "unknown"
	}
}

func exprString(fset *token.FileSet, expr ast.Expr) string {
	var buf bytes.Buffer
	format.Node(&buf, fset, expr)
	lines := strings.Split(buf.String(), "\n")
	result := make([]string, 0, len(lines))
	for _, l := range lines {
		result = append(result, strings.TrimSpace(l))
	}
	return strings.Join(result, `\n`)
}
 := make([]string, 0, len(lines))
	for _, l := range lines {
		result = append(result, strings.TrimSpace(l))
	}
	return strings.Join(result, `\n`)
}
