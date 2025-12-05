// app/(tabs)/predict.tsx
import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';

export default function PredictScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Market Predict</Text>
      <Text style={styles.subtitle}>Tap a card to predict</Text>

      {/* GOLD CARD - Clickable */}
      <Pressable
        style={styles.card}
        onPress={() => router.push('/predict/gold')}
      >
        <Text style={styles.cardTitle}>GOLD / USD</Text>
        <Text style={styles.price}>$2,420.50</Text>
        <Text style={styles.trend}>↑ 2.1% today</Text>
      </Pressable>
      

      {/* Add more cards later: BTC, Oil, etc. */}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f1e', padding: 20 },
  title: { fontSize: 32, fontWeight: '800', color: '#fff', marginTop: 20 },
  subtitle: { fontSize: 16, color: '#888', marginBottom: 30 },
  card: {
    backgroundColor: '#1a1a2e',
    padding: 24,
    borderRadius: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#333',
  },
  cardTitle: { color: '#00d4ff', fontSize: 16, fontWeight: '700' },
  price: { color: '#fff', fontSize: 32, fontWeight: 'bold', marginVertical: 8 },
  trend: { color: '#4ade80', fontSize: 16 },
});