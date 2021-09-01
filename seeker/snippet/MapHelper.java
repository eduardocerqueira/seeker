//date: 2021-09-01T13:14:36Z
//url: https://api.github.com/gists/c3306b372ef8f0ad78cbfb45efd48898
//owner: https://api.github.com/users/TheDoctorOne

public class MapHelper {
	public static <T extends Enum, V> Map<T, V> fromOrdinal(Map<Integer, V> ordMap, Class<T> enumClass) {
		Map<T, V> org = new HashMap<>();
		for(int ord : ordMap.keySet()) {
			org.put(enumClass.getEnumConstants()[ord], ordMap.get(ord));
		}
		return org;
	}
	
	public static <T extends Enum, V> Map<Integer, V> toOrdinal(Map<T, V> map) {
		Map<Integer, V> ord = new HashMap<>();
		for(T key : map.keySet()) {
			ord.put(key.ordinal(), map.get(key));
		}
		return ord;
	}
}