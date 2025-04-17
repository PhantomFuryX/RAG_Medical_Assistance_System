import psutil

def get_cpu_info():
    """
    Returns information about CPU utilization and system details, including P-core and E-core utilization.
    """
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    
    # Example logic: Assume the first half are P-cores and the second half are E-cores
    # Adjust this based on your system's architecture
    p_core_count = cpu_count // 2
    e_core_count = cpu_count - p_core_count
    
    p_core_utilization = cpu_percent[:p_core_count]
    e_core_utilization = cpu_percent[p_core_count:]
    
    return {
        "cpu_count": cpu_count,
        "cpu_frequency": {
            "current": cpu_freq.current,
            "min": cpu_freq.min,
            "max": cpu_freq.max
        },
        "p_core_count": p_core_count,
        "e_core_count": e_core_count,
        "p_core_utilization": p_core_utilization,
        "e_core_utilization": e_core_utilization,
        "message": f"CPU has {cpu_count} cores with a current frequency of {cpu_freq.current:.2f} MHz."
    }

def print_cpu_info():
    """
    Prints information about CPU utilization and system details, including P-core and E-core utilization.
    """
    info = get_cpu_info()
    
    print(f"🖥️ {info['message']}")
    print(f"  - Frequency: {info['cpu_frequency']['current']:.2f} MHz (Min: {info['cpu_frequency']['min']:.2f}, Max: {info['cpu_frequency']['max']:.2f})")
    print(f"  - P-Cores ({info['p_core_count']}):")
    for i, percent in enumerate(info["p_core_utilization"]):
        print(f"    P-Core {i}: {percent:.2f}%")
    print(f"  - E-Cores ({info['e_core_count']}):")
    for i, percent in enumerate(info["e_core_utilization"]):
        print(f"    E-Core {i}: {percent:.2f}%")
