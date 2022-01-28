//date: 2022-01-28T16:52:19Z
//url: https://api.github.com/gists/3fc3e1039674e9059289153da174f381
//owner: https://api.github.com/users/tonvanbart

package io.axual.connect.plugins.kafka;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class Tryout {

    @Test
    void testEnumValues() {
        Thing thing = Thing.fromType("fooBar");
        assertEquals("hello from fooBar", thing.work());

        assertEquals("hello from else", Thing.OTHER.work());
        assertEquals("a horse with no name", Thing.UNNAMED.work());

        assertThrows(IllegalArgumentException.class, () -> Thing.fromType("nonsense"));

        assertEquals("hello from else", Thing.fromType("else").plainMethod());
    }

    interface Canwork {
        String work();
    }

    enum Thing implements Canwork {
        FOO("fooBar"),
        OTHER("else"),
        UNNAMED("") {
            @Override
            public String work() {
                return "a horse with no name";
            }
        };

        private final Canwork worker;

        Thing(String type) {
            this.worker = () -> "hello from " + type;
        }

        static Thing fromType(String typeName) {
            if ("fooBar".equals(typeName)) {
                return FOO;
            } else if("else".equals(typeName)) {
                return OTHER;
            } else {
                throw new IllegalArgumentException("unknown type name");
            }
        }

        @Override
        public String work() {
            return worker.work();
        }

        public String plainMethod() {
            return worker.work();
        }
    }
}
