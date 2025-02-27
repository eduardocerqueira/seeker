//date: 2025-02-27T16:57:57Z
//url: https://api.github.com/gists/99ced7402b3082513f81be809c4b01a6
//owner: https://api.github.com/users/pacphi

package me.pacphi.ai.resos.csv;

public interface EntityMapper<T> {
    T mapFromCsv(String[] line) throws CsvMappingException;
    Class<T> getEntityClass();
}