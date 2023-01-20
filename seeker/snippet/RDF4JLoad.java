//date: 2023-01-20T16:45:56Z
//url: https://api.github.com/gists/cf2b9429eeefc5a61cd28fe24d1cfac1
//owner: https://api.github.com/users/cbuil

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.eclipse.rdf4j.common.exception.RDF4JException;
import org.eclipse.rdf4j.common.transaction.IsolationLevels;
import org.eclipse.rdf4j.repository.Repository;
import org.eclipse.rdf4j.repository.RepositoryConnection;
import org.eclipse.rdf4j.repository.sail.SailRepository;
import org.eclipse.rdf4j.rio.RDFFormat;
import org.eclipse.rdf4j.sail.lmdb.LmdbStore;
import org.eclipse.rdf4j.sail.lmdb.config.LmdbStoreConfig;

public class RDF4JLoad {
	private static String WIKIDATA_URI = "http://wikidata.org/";

	public static void main(String[] args) {
		File dataDir = new File(args[0]);
		LmdbStoreConfig config = new LmdbStoreConfig();
		// set triple indexes
		config.setTripleIndexes("spoc,ospc,psoc");
		// set max db sizes to 1 TiB each
		config.setValueDBSize(1_099_511_627_776L);
		config.setTripleDBSize(1_099_511_627_776L);

		LmdbStore sail = new LmdbStore(dataDir, config);
		Repository repo = new SailRepository(sail);
		RepositoryConnection connection = repo.getConnection();
		try {
			long start = System.currentTimeMillis();
			try {
				int numFiles = 0;
				int numErrorFiles = 0;
				List<String> errorFiles = new ArrayList<String>();
				List<File> triplesFileList = Files.walk(Paths.get(args[1])).filter(Files::isRegularFile)
						.map(Path::toFile).collect(Collectors.toList());
				System.out.println("Number of files: " + triplesFileList.size());
				connection.begin(IsolationLevels.NONE);
				for (File triplesFile : triplesFileList) {
					long fileStart = System.currentTimeMillis();
					// add code here to load your data
					try {
						connection.add(triplesFile, WIKIDATA_URI, RDFFormat.NTRIPLES);
					} catch (Exception e) {
						System.out.println("Partsing error: " + e.getMessage() + " in file " + triplesFile.getName());
						errorFiles.add(triplesFile.getName());
						numErrorFiles += 1;
					}
					connection.commit();
					long duration = System.currentTimeMillis() - fileStart;
					numFiles += 1;
					System.out.println("loaded " + triplesFile.toString() + " in " + (duration / 1000.0)
							+ " seconds. File " + numFiles);
				}
				long duration = System.currentTimeMillis() - start;
				System.out.println("loaded " + numFiles + " in " + (duration / 1000.0) + " seconds ");
				System.out.println("NOT loaded " + numErrorFiles);
				System.out.println("Error files: ");
				for (String errorFile : errorFiles) {
					System.out.println(errorFile);
				}
			} finally {
				System.out.println("Closing connection...");
				connection.close();
			}
		} catch (RDF4JException e) {
			e.printStackTrace();
		} catch (java.io.IOException e) {
			e.printStackTrace();
		}

	}
}
