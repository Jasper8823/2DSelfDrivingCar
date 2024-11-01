using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(NNet))]

public class Ride : MonoBehaviour
{
    private Rigidbody2D _rigidbody;
    [SerializeField] private float _force;
    [SerializeField] private float _torque;

    private NNet network;

    [SerializeField] Transform _castP;

    private Vector3 lastPos, startPos, startRot;


    [Header("Network Options")]
    public int LAYERS = 1;
    public int NEURONS = 10;

    [Range(-1f, 1f)]
    public float a, t;

    public float timeStart = 0f;

    [Header("Fitness")]
    public float overall;

    public float distMul = 1.4f;
    public float speedMul = 0.6f;
    public float sensorMul = 0f;

    private float distTrav;
    private float speedAvg;

    [Header("Sensors")]
    private float aSensor, bSensor, cSensor, dSensor, eSensor;

    private void Awake()
    {
        _rigidbody = GetComponent<Rigidbody2D>();
        startPos = transform.position;
        startRot = transform.eulerAngles;

        network = GetComponent<NNet>();

    }

    public void CalculateFitness()
    {
        distTrav += Vector3.Distance(transform.position, lastPos);
        speedAvg = distTrav / timeStart;

        overall = (distTrav * distMul) + (speedAvg * speedMul) + ((aSensor + bSensor + cSensor + dSensor + eSensor) / 5 * sensorMul);

        if (timeStart > 20 && overall < 20)
        {
            Death();
        }
        if (overall >= 1000)
        {
            Death();
        }
    }

    public void InputSensors()
    {
        Debug.DrawRay(_castP.position, Vector3.up, Color.green);
        RaycastHit2D hit = Physics2D.Raycast(_castP.position, _castP.rotation*-Vector3.right);
        if (hit.collider != null)
        {
            aSensor = hit.distance / 20;
        }

        hit = Physics2D.Raycast(_castP.position, _castP.rotation * (Vector3.up-Vector3.right));
        if (hit.collider != null)
        {
            bSensor = hit.distance / 20;
        }

        hit = Physics2D.Raycast(_castP.position, _castP.rotation * Vector3.up);
        if (hit.collider != null)
        {
            cSensor = hit.distance / 20;
        }

        hit = Physics2D.Raycast(_castP.position, _castP.rotation * (Vector3.right + Vector3.up));
        if (hit.collider != null)
        {
            dSensor = hit.distance / 20;
        }

        hit = Physics2D.Raycast(_castP.position, _castP.rotation * Vector3.right);
        if (hit.collider != null)
        {
            eSensor = hit.distance / 20;
        }
    }

    private void Death()
    {
        GameObject.FindObjectOfType<GeneticManager>().Death(overall, network);
    }

    public void ResetWithNetwork(NNet net)
    {
        network = net;
        Reset();
    }

    public void Reset()
    {
        timeStart = 0f;
        speedAvg = 0f;
        distTrav = 0f;
        overall = 0f;
        transform.position = startPos;
        transform.eulerAngles = startRot;
        lastPos = startPos;
        _rigidbody.angularVelocity = 0f;
        _rigidbody.velocity = Vector2.zero;
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        Death();
    }

    void Start()
    {
        lastPos = transform.position;
    }

    public void MoveCar(float v, float h)
    {
        //Vector2 inp = Vector2.Lerp(Vector2.zero, new Vector2(0, v * 11.4f), 0.02f);
        //inp = transform.TransformDirection(inp);

        //_rigidbody.MovePosition(_rigidbody.position + inp);
        if (Vector3.Distance(transform.position, lastPos) > 0.025f)
        {
            _rigidbody.AddTorque(h * _torque);
        }
        _rigidbody.AddRelativeForce(Vector2.up * v * _force, ForceMode2D.Impulse);
        //float rotation = h * 90 * 0.02f;
        //_rigidbody.MoveRotation(_rigidbody.rotation + rotation);
    }

    void FixedUpdate()
    {
        /*if (Input.GetKey(KeyCode.W))
        {
            _rigidbody.AddRelativeForce(Vector2.up * _force, ForceMode2D.Impulse);
        }
        if (Vector3.Distance(transform.position, lastPos) > 0.025f && Input.GetKey(KeyCode.D))
        {
            _rigidbody.AddTorque(-_torque);
        }
        else if (Vector3.Distance(transform.position, lastPos) > 0.025f && Input.GetKey(KeyCode.A))
        {
            _rigidbody.AddTorque(_torque);
        }*/
        InputSensors();
        timeStart += Time.deltaTime;

        (a, t) = network.RunNetwork(aSensor, bSensor, cSensor, dSensor, eSensor);

        MoveCar(a, t);

        CalculateFitness();
        lastPos = transform.position;
    }
}
