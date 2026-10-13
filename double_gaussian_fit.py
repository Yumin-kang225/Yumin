#과제
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 데이터 불러오기
data = np.loadtxt("hist2.csv", delimiter=",")

x = data[:, 0]
y = data[:, 1]


# 2. 가우시안 함수 정의
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


# 3. 가우시안 함수 두 개의 합 정의
def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return (gaussian(x, A1, mu1, sigma1)
            + gaussian(x, A2, mu2, sigma2))


# 4. 피팅 수행
initial_guess = [1100, 1.0, 0.25, 550, 2.0, 0.5]

popt, pcov = curve_fit(
    double_gaussian,
    x,
    y,
    p0=initial_guess,
    bounds=(
        [0, 0.5, 0.01, 0, 1.3, 0.01],
        [2000, 1.4, 1.0, 1500, 2.8, 1.5]
    ),
    maxfev=20000
)

A1, mu1, sigma1, A2, mu2, sigma2 = popt


# 5. 각 입자의 가우시안 함수 계산
fit_A = gaussian(x, A1, mu1, sigma1)
fit_B = gaussian(x, A2, mu2, sigma2)
fit_total = fit_A + fit_B


# 6. 각 가우시안 함수의 적분값 계산
integral_A = A1 * sigma1 * np.sqrt(2 * np.pi)
integral_B = A2 * sigma2 * np.sqrt(2 * np.pi)

total_integral = integral_A + integral_B

ratio_A = integral_A / total_integral * 100
ratio_B = integral_B / total_integral * 100


# 7. 피팅 결과 출력
print("입자 A")
print(f"A = {A1:.2f}, mu = {mu1:.4f}, sigma = {sigma1:.4f}")
print(f"적분값 = {integral_A:.2f}")
print(f"생성 비율 = {ratio_A:.2f}%")

print("\n입자 B")
print(f"A = {A2:.2f}, mu = {mu2:.4f}, sigma = {sigma2:.4f}")
print(f"적분값 = {integral_B:.2f}")
print(f"생성 비율 = {ratio_B:.2f}%")


# 8. 그래프 그리기
plt.figure(figsize=(10, 6))

plt.scatter(x, y, color="black", s=20, label="Measured data")
plt.plot(x, fit_total, color="red", label="Total fit")
plt.plot(x, fit_A, "--", label="Particle A")
plt.plot(x, fit_B, "--", label="Particle B")

plt.xlabel("Energy")
plt.ylabel("Count")
plt.title("Double Gaussian Fit")
plt.legend()
plt.grid()
plt.show()