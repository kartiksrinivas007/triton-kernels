grep -E 'RestrictProfiling|RmProfiling' /proc/driver/nvidia/params
# enable at module load
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-perf.conf

# rebuild initramfs so the option persists across reboots
sudo update-initramfs -u

# reboot to reload the nvidia kernel module with the new param
sudo reboot