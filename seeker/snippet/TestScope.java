//date: 2022-02-16T17:08:28Z
//url: https://api.github.com/gists/c1c0942efcd3950c8691c0863fb9cd99
//owner: https://api.github.com/users/Saifuddin-Shaikh

@SpringBootTest
public class TestScope {

	@Autowired
	Consumer1 consumer1;
	@Autowired
	Consumer2 consumer2;
	
	@Test
	void testSingleton() throws Exception {
		ScopeBeanExample bean1 = consumer1.getScopeBeanExample();
		ScopeBeanExample bean2 = consumer2.getScopeBeanExample();
		System.out.println(bean1.getName());
		System.out.println(bean2.getName());
		assertEquals(bean1, bean2);
	}

}
