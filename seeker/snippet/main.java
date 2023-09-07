//date: 2023-09-07T16:52:34Z
//url: https://api.github.com/gists/2d9ca5773a1958cc7a4f5cc7b3faa8d4
//owner: https://api.github.com/users/vdparikh

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Newtonsoft.Json;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace KeyValueStoreAPI
{
    public class KeyValueStore
    {
        private readonly ConcurrentDictionary<string, string> _store = new ConcurrentDictionary<string, string>();

        public string Get(string key)
        {
            _store.TryGetValue(key, out string value);
            return value;
        }

        public void Set(string key, string value)
        {
            _store[key] = value;
        }

        public bool Update(string key, string value)
        {
            if (_store.ContainsKey(key))
            {
                _store[key] = value;
                return true;
            }
            return false;
        }

        public bool Delete(string key)
        {
            return _store.TryRemove(key, out _);
        }

        public IEnumerable<string> GetAllKeys()
        {
            return _store.Keys;
        }
    }

    public class Startup
    {
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddSingleton<KeyValueStore>();
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            app.UseRouting();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapGet("/get", async context =>
                {
                    var key = context.Request.Query["key"];
                    var keyValueStore = context.RequestServices.GetRequiredService<KeyValueStore>();
                    var value = keyValueStore.Get(key);
                    if (value != null)
                    {
                        await context.Response.WriteAsync($"Key: {key}, Value: {value}");
                    }
                    else
                    {
                        context.Response.StatusCode = 404;
                    }
                });

                endpoints.MapPost("/set", async context =>
                {
                    var data = await ReadRequestBody(context);
                    var key = data["key"];
                    var value = data["value"];
                    var keyValueStore = context.RequestServices.GetRequiredService<KeyValueStore>();
                    keyValueStore.Set(key, value);
                    await context.Response.WriteAsync($"Key: {key}, Value: {value} is set");
                });

                endpoints.MapPut("/update", async context =>
                {
                    var data = await ReadRequestBody(context);
                    var key = data["key"];
                    var value = data["value"];
                    var keyValueStore = context.RequestServices.GetRequiredService<KeyValueStore>();
                    var updated = keyValueStore.Update(key, value);
                    if (updated)
                    {
                        await context.Response.WriteAsync($"Key: {key}, Value: {value} is updated");
                    }
                    else
                    {
                        context.Response.StatusCode = 404;
                    }
                });

                endpoints.MapDelete("/delete", async context =>
                {
                    var key = context.Request.Query["key"];
                    var keyValueStore = context.RequestServices.GetRequiredService<KeyValueStore>();
                    var deleted = keyValueStore.Delete(key);
                    if (deleted)
                    {
                        await context.Response.WriteAsync($"Key: {key} is deleted");
                    }
                    else
                    {
                        context.Response.StatusCode = 404;
                    }
                });

                endpoints.MapGet("/list", async context =>
                {
                    var keyValueStore = context.RequestServices.GetRequiredService<KeyValueStore>();
                    var keys = keyValueStore.GetAllKeys();
                    await context.Response.WriteAsync(JsonConvert.SerializeObject(keys));
                });
            });
        }

        private static async Task<Dictionary<string, string>> ReadRequestBody(HttpContext context)
        {
            try
            {
                var body = await new StreamReader(context.Request.Body).ReadToEndAsync();
                return JsonConvert.DeserializeObject<Dictionary<string, string>>(body);
            }
            catch (Exception)
            {
                return new Dictionary<string, string>();
            }
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            CreateHostBuilder(args).Build().Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });
    }
}
