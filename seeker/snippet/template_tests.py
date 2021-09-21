#date: 2021-09-21T16:53:13Z
#url: https://api.github.com/gists/be9a34e97d8903ccd52701f367b61802
#owner: https://api.github.com/users/KoliosterNikolayIliev

import unittest

from template import TemplateEngine, TemplateEngineError


class TemplateEngineTests(unittest.TestCase):
    """
    Should a TemplateError be raised if single curly brackets are used ("{} instead of {{}} inside the template")?
    """

    def tests_TemplateEngine(self):
        with self.subTest('Single line even amount of whitespaces'):
            template = "Hello there, {{ x }}"
            engine = TemplateEngine(template)
            self.assertEqual(engine.render(x="general Kenobi."), "Hello there, general Kenobi.")

        with self.subTest('Single line no whitespaces'):
            template = "Hello there, {{x}}"
            engine = TemplateEngine(template)
            self.assertEqual(engine.render(x="general Kenobi."), "Hello there, general Kenobi.")

        with self.subTest('Single whitespace on Left'):
            template = "Hello there, {{ x}}"
            engine = TemplateEngine(template)
            self.assertEqual(engine.render(x="general Kenobi."), "Hello there, general Kenobi.")

        with self.subTest('Single whitespace on right'):
            template = "Hello there, {{x }}"
            engine = TemplateEngine(template)
            self.assertEqual(engine.render(x="general Kenobi."), "Hello there, general Kenobi.")

        with self.subTest('Multiple whitespaces both sides'):
            template = "Hello there, {{    x    }}"
            engine = TemplateEngine(template)
            self.assertEqual(engine.render(x="general Kenobi."), "Hello there, general Kenobi.")

        with self.subTest('Multiple whitespaces left'):
            template = "Hello there, {{    x}}"
            engine = TemplateEngine(template)
            self.assertEqual(engine.render(x="general Kenobi."), "Hello there, general Kenobi.")

        with self.subTest('Multiple whitespaces right'):
            template = "Hello there, {{x    }}"
            engine = TemplateEngine(template)
            self.assertEqual(engine.render(x="general Kenobi."), "Hello there, general Kenobi.")

        """
        """
        with self.subTest('Multiline even amount of whitespaces'):
            template = """
            Hello {{ first_name }} {{ last_name }},

            I hope this email finds you well.

            We are currently running a promotion for {{ product }}.

            You can get your discount {{ here }}
            """
            engine = TemplateEngine(template)
            result = engine.render(first_name="Ivan",
                                   last_name="Ivanov",
                                   product="Python course",
                                   here="https://hackbulgaria.com/python-101-forever")

            expected = """
            Hello Ivan Ivanov,

            I hope this email finds you well.

            We are currently running a promotion for Python course.

            You can get your discount https://hackbulgaria.com/python-101-forever
            """

            self.assertMultiLineEqual(result, expected)

            with self.subTest(' Multiline even amount of whitespaces excluding white spaces when comparing'):
                template = """
                Hello {{ first_name }} {{ last_name }},

                I hope this email finds you well.

                We are currently running a promotion for {{ product }}.

                You can get your discount {{ here }}
                """
                engine = TemplateEngine(template)
                result = engine.render(first_name="Ivan",
                                       last_name="Ivanov",
                                       product="Python course",
                                       here="https://hackbulgaria.com/python-101-forever")

                result = [x.strip() for x in result if x != '']

                expected = """
                Hello Ivan Ivanov,

                I hope this email finds you well.

                We are currently running a promotion for Python course.

                You can get your discount https://hackbulgaria.com/python-101-forever
                """
                expected = [x.strip() for x in expected if x != '']

                self.assertEqual(result, expected)

        with self.subTest('Multiline uneven amount of whitespaces `left right`'):
            template = """
            Hello {{first_name }} {{ last_name}},

            I hope this email finds you well.

            We are currently running a promotion for {{          product }}.

            You can get your discount {{ here           }}
            """
            engine = TemplateEngine(template)
            result = engine.render(first_name="Ivan",
                                   last_name="Ivanov",
                                   product="Python course",
                                   here="https://hackbulgaria.com/python-101-forever")

            expected = """
            Hello Ivan Ivanov,

            I hope this email finds you well.

            We are currently running a promotion for Python course.

            You can get your discount https://hackbulgaria.com/python-101-forever
            """

            self.assertMultiLineEqual(result, expected)

        with self.subTest('Context not matching all variables Raise TemplateEngineError'):
            template = """
            Hello {{ first_name }} {{ last_name }},

            I hope this email finds you well.

            We are currently running a promotion for {{ product }}.

            You can get your discount {{ here }}
            """
            engine = TemplateEngine(template)

            with self.assertRaises(TemplateEngineError,
                                   msg='Not all variables, present in `template`, have values in `context`'):
                engine.render(
                    last_name="Ivanov",
                    product="Python course",
                    here="https://hackbulgaria.com/python-101-forever"
                )

        with self.subTest('VariablesList'):
            with self.subTest('Single variable'):
                template = "Hello there, {{ x }}"
                engine = TemplateEngine(template)
                self.assertEqual(engine.extract_variables(), ['x'])
            with self.subTest('Multiple variables'):
                multivar_template = """
                Hello {{ first_name }} {{ last_name }},

                I hope this email finds you well.

                We are currently running a promotion for {{ product }}.

                You can get your discount {{ here }}
                """
                engine_multi = TemplateEngine(multivar_template)
                self.assertEqual(engine_multi.extract_variables(), ['first_name', 'last_name', 'product', 'here'])


if __name__ == '__main__':
    unittest.main()