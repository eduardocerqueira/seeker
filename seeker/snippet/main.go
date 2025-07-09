//date: 2025-07-09T16:54:13Z
//url: https://api.github.com/gists/9464208c774ddbd294c6734a9c8f9eba
//owner: https://api.github.com/users/itskenzylimon

package main

import (
	"Scalable-Secure-Go-Web/internal/config"
	"Scalable-Secure-Go-Web/internal/models"
	"fmt"
	"log"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/helmet"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func main() {
	// 1️⃣ Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	// 2️⃣ Setup DB (SQLite for demo; swap for Postgres/MySQL in prod)
	db, err := gorm.Open(sqlite.Open("catalog.db"), &gorm.Config{})
	if err != nil {
		log.Fatalf("database error: %v", err)
	}
	if err := db.AutoMigrate(&models.Product{}, &models.Brand{}, &models.Category{}); err != nil {
		log.Fatalf("migration error: %v", err)
	}

	// 3️⃣ Initialize Fiber
	app := fiber.New()

	// 4️⃣ Global middleware
	app.Use(logger.New())  // “Yes boss, we LOG stuff.”
	app.Use(recover.New()) // Panic–proof the thing.

	// 4.1️⃣ CORS – only if origins are configured
	if len(cfg.FrontendOrigins) > 0 {
		app.Use(cors.New(cors.Config{
			AllowOrigins:     join(cfg.FrontendOrigins, ","),
			AllowMethods:     "GET,POST,PUT,PATCH,DELETE,OPTIONS",
			AllowHeaders:     "Origin, Content-Type, Accept, Authorization",
			AllowCredentials: cfg.CORSAllowCreds,
		}))
	}

	// 4.2️⃣ Helmet – optional toggle
	if cfg.EnableHelmet {
		app.Use(helmet.New())
	}

	// 4.3️⃣ Rate limiter – optional toggle
	if cfg.EnableLimiter {
		app.Use(limiter.New(limiter.Config{
			Max:          cfg.RateLimitMax,
			Expiration:   cfg.RateLimitWindow,
			LimitReached: tooManyRequestsJSON,
		}))
	}

	// 5️⃣ Dummy route to prove it works
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok"})
	})

	// 6️⃣ Start server
	addr := fmt.Sprintf(":%d", cfg.Port)
	log.Printf("⇨ Listening on %s", addr)
	if err := app.Listen(addr); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// tooManyRequestsJSON returns 429 with JSON message.
func tooManyRequestsJSON(c *fiber.Ctx) error {
	return c.Status(fiber.StatusTooManyRequests).
		JSON(fiber.Map{"error": "Too many requests. Calm down, champ."})
}

// join is a tiny helper (because strings.Join needs []string anyway)
func join(ss []string, sep string) string {
	if len(ss) == 0 {
		return ""
	}
	out := ss[0]
	for _, s := range ss[1:] {
		out += sep + s
	}
	return out
}
