#date: 2024-10-01T16:57:47Z
#url: https://api.github.com/gists/b1db8af86229380b16ef05c4d9a27f3e
#owner: https://api.github.com/users/zzpzaf

FROM arm64v8/debian:bookworm
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=eclipse-temurin:17 $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor && mkdir -p /var/log/supervisor

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy the script that monitors and runs the most recent jar and make it executable
COPY run_jar.sh /usr/local/bin/run_jar.sh
RUN chmod +x /usr/local/bin/run_jar.sh

# Expose port 8080 to other containers within the same Docker network (no host mapping)
EXPOSE 8080  

# Set the entrypoint to Supervisor
CMD ["/usr/bin/supervisord"]
