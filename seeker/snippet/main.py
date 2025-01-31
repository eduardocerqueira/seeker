#date: 2025-01-31T16:41:37Z
#url: https://api.github.com/gists/55f4cc650775f7bf1645abe3bf811d3e
#owner: https://api.github.com/users/markusand

def main():
    """Main function"""
    gga = GGA.parse("$GPGGA,202530.00,5109.0262,N,11401.8407,W,5,40,0.5,1097.36,M,-17.00,M,18,NRTI*61")
    print(gga)
    print(f"Lat:{gga.lat:.6f} Lon:{gga.lon:.6f}, Alt:{gga.alt:.1f}m")


if __name__ == "__main__":
    main()