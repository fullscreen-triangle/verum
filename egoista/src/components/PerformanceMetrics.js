// components/PerformanceMetrics.js
import { useRef } from "react";
import { useInView } from "framer-motion";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  LogarithmicScale
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function PerformanceMetrics() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  // Prediction Error Growth (Lyapunov Divergence)
  const predictionData = {
    labels: ['0s', '0.5s', '1.0s', '1.5s', '2.0s', '2.5s', '3.0s'],
    datasets: [
      {
        label: 'Prediction Error (meters)',
        data: [0.1, 0.16, 0.27, 0.45, 0.74, 1.22, 2.01],
        borderColor: 'rgb(220, 38, 38)',
        backgroundColor: 'rgba(220, 38, 38, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Safety Threshold',
        data: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        borderColor: 'rgb(234, 179, 8)',
        borderDash: [5, 5],
        pointRadius: 0
      }
    ]
  };

  // Computational Complexity Comparison
  const complexityData = {
    labels: ['100', '500', '1K', '5K', '10K', '50K', '100K'],
    datasets: [
      {
        label: 'Forward Simulation O(N)',
        data: [100, 500, 1000, 5000, 10000, 50000, 100000],
        borderColor: 'rgb(220, 38, 38)',
        backgroundColor: 'rgba(220, 38, 38, 0.1)'
      },
      {
        label: 'A* Navigation O(N log N)',
        data: [200, 1350, 3000, 21500, 46000, 282000, 600000],
        borderColor: 'rgb(234, 179, 8)',
        backgroundColor: 'rgba(234, 179, 8, 0.1)'
      },
      {
        label: 'Backward Completion O(log₃ N)',
        data: [4.2, 5.7, 6.3, 7.5, 8.4, 9.9, 10.5],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)'
      }
    ]
  };

  const predictionOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'rgb(156, 163, 175)',
          font: { size: 14 }
        }
      },
      title: {
        display: true,
        text: 'Prediction Error Growth (λ = 1.0 s⁻¹)',
        color: 'rgb(209, 213, 219)',
        font: { size: 18, weight: 'bold' }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Error (meters)',
          color: 'rgb(156, 163, 175)'
        },
        ticks: { color: 'rgb(156, 163, 175)' },
        grid: { color: 'rgba(156, 163, 175, 0.1)' }
      },
      x: {
        title: {
          display: true,
          text: 'Time (seconds)',
          color: 'rgb(156, 163, 175)'
        },
        ticks: { color: 'rgb(156, 163, 175)' },
        grid: { color: 'rgba(156, 163, 175, 0.1)' }
      }
    }
  };

  const complexityOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'rgb(156, 163, 175)',
          font: { size: 14 }
        }
      },
      title: {
        display: true,
        text: 'Computational Complexity Comparison',
        color: 'rgb(209, 213, 219)',
        font: { size: 18, weight: 'bold' }
      }
    },
    scales: {
      y: {
        type: 'logarithmic',
        title: {
          display: true,
          text: 'Operations (log scale)',
          color: 'rgb(156, 163, 175)'
        },
        ticks: { color: 'rgb(156, 163, 175)' },
        grid: { color: 'rgba(156, 163, 175, 0.1)' }
      },
      x: {
        title: {
          display: true,
          text: 'Network Size (nodes)',
          color: 'rgb(156, 163, 175)'
        },
        ticks: { color: 'rgb(156, 163, 175)' },
        grid: { color: 'rgba(156, 163, 175, 0.1)' }
      }
    }
  };

  return (
    <div ref={ref} className="my-32">
      <h2 className="font-bold text-8xl mb-16 w-full text-center md:text-6xl xs:text-4xl md:mb-8">
        Performance Analysis
      </h2>
      
      <div className="grid grid-cols-2 gap-12 lg:grid-cols-1 lg:gap-8">
        {/* Prediction Error Chart */}
        <div className={`relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark transition-all duration-700 ${
          isInView ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-10'
        }`}>
          <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
          <div style={{ height: '400px' }}>
            <Line data={predictionData} options={predictionOptions} />
          </div>
          <p className="mt-4 text-sm text-dark/75 dark:text-light/75 font-medium">
            Exponential error growth ε(t) = ε₀e^(λt) renders prediction useless beyond 2.3 seconds. 
            Conventional AVs require 5–15 second horizons—mathematically impossible.
          </p>
        </div>

        {/* Complexity Comparison Chart */}
        <div className={`relative rounded-2xl border-2 border-solid border-dark bg-light p-8 dark:border-light dark:bg-dark transition-all duration-700 ${
          isInView ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-10'
        }`} style={{ transitionDelay: '200ms' }}>
          <div className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl bg-dark dark:bg-light" />
          <div style={{ height: '400px' }}>
            <Line data={complexityData} options={complexityOptions} />
          </div>
          <p className="mt-4 text-sm text-dark/75 dark:text-light/75 font-medium">
            Backward completion achieves O(log₃ N) complexity—10⁵× faster than A* on 100K-node networks. 
            Performance improves as network size increases (counterintuitive!).
          </p>
        </div>
      </div>
    </div>
  );
}
