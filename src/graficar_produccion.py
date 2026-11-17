import pandas as pd
import matplotlib.pyplot as plt

archivo = "data/cafe_valle_limpio.csv"


df = pd.read_csv(archivo)


cartago = df[df["codigo_municipio"] == 76147].copy()


cartago = cartago.sort_values("año")


plt.figure(figsize=(10, 6))

plt.plot(
    cartago["año"],
    cartago["produccion"],
    marker="o",
    linewidth=2
)

plt.title("Producción de Café en Cartago, Valle del Cauca")
plt.xlabel("Año")
plt.ylabel("Producción")
plt.grid(True, alpha=0.3)


for _, fila in cartago.iterrows():
    plt.annotate(
        f"{fila['produccion']:.2f}",
        (fila["año"], fila["produccion"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center"
    )

plt.tight_layout()

salida = "data/produccion_cartago.png"
plt.savefig(salida, dpi=300)

print(f"Gráfica guardada en: {salida}")

plt.show()