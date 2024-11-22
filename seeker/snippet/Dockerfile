#date: 2024-11-22T16:48:51Z
#url: https://api.github.com/gists/0146020f97fb5ea8249534bc92bcd69b
#owner: https://api.github.com/users/lbussell

FROM scratch AS program
COPY <<EOF /Program.cs
using System;
using static System.Console;
// foreach (TimeZoneInfo z in TimeZoneInfo.GetSystemTimeZones())
// {
//     WriteLine(z.Id);
// }
TimeZoneInfo z1 = TimeZoneInfo.FindSystemTimeZoneById("Europe/Kyiv");
WriteLine(z1.Id);
TimeZoneInfo z2 = TimeZoneInfo.FindSystemTimeZoneById("FLE Standard Time");
WriteLine(z2.Id);
EOF

FROM mcr.microsoft.com/dotnet/sdk:8.0-jammy AS jammy-build
RUN dotnet new console -n App -o /src --no-restore
WORKDIR /src
COPY --link --from=program /Program.cs /src/Program.cs
RUN dotnet restore
RUN dotnet publish --no-restore -o /app

FROM mcr.microsoft.com/dotnet/sdk:8.0-noble AS noble-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata-icu tzdata-legacy \
    && rm -rf /var/lib/apt/lists/*
RUN dotnet new console -n App -o /src --no-restore
WORKDIR /src
COPY --link --from=program /Program.cs /src/Program.cs
RUN dotnet restore
RUN dotnet publish --no-restore -o /app

FROM mcr.microsoft.com/dotnet/runtime:8.0-jammy AS jammy-runtime
COPY --link --from=jammy-build /app /app/
ENTRYPOINT ["/app/App"]

FROM mcr.microsoft.com/dotnet/runtime:8.0-noble AS noble-runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata-icu tzdata-legacy \
    && rm -rf /var/lib/apt/lists/*
COPY --link --from=noble-build /app /app/
ENTRYPOINT ["/app/App"]
